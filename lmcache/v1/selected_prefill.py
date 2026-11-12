# SPDX-License-Identifier: Apache-2.0
"""Allocator ownership for optional reads of a prepared prefill source."""

# Standard
from collections.abc import Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Protocol

# First Party
from lmcache.v1.memory_management import (
    LayerPageSource,
    MemoryObj,
    TensorMemoryObj,
)


class PrefillReadCompletion(Protocol):
    """A completion event covering every asynchronous reader of a step."""

    def synchronize(self) -> None:
        """Return after all covered reads have completed."""


@dataclass(frozen=True, slots=True)
class SelectedPrefillRequest:
    """The exact half-open token interval requested from a main KV source."""

    request_id: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not self.request_id or not 0 <= self.start < self.end:
            raise ValueError("Invalid selected prefill request interval")


@dataclass(frozen=True, slots=True)
class SelectedPrefillSource:
    """Physical host objects and token coverage, in layer and chunk order."""

    request: SelectedPrefillRequest
    spans: tuple[tuple[int, int], ...]
    layers: tuple[LayerPageSource | tuple[MemoryObj, ...], ...]


class SelectedPrefillStep:
    """Single-use handoff for main KV objects retrieved by a prefill step.

    The connector stages sources before submitting its initial bank window.
    One caller can acquire them after the connector completes setup and must
    retire the step with a completion event covering all asynchronous readers.
    A step with no acquired reader can be retired without an event.
    """

    def __init__(self, requests: Sequence[SelectedPrefillRequest]) -> None:
        self._requests = {request.request_id: request for request in requests}
        if len(self._requests) != len(requests):
            raise ValueError("Duplicate request IDs in selected prefill step")
        self._sources: dict[str, SelectedPrefillSource] = {}
        self._owners: dict[int, TensorMemoryObj] = {}
        self._lock = Lock()
        self._state = "new"

    def requests_source(self, request_id: str) -> bool:
        """Return whether this step was configured to capture the request."""
        return request_id in self._requests

    def begin(self) -> None:
        """Reserve a fresh handle for connector setup."""
        with self._lock:
            if self._state != "new":
                raise RuntimeError("Selected prefill step is already used")
            self._state = "staging"

    def stage(
        self,
        request_id: str,
        starts: Sequence[int],
        ends: Sequence[int],
        layers: Sequence[LayerPageSource | Sequence[MemoryObj] | None],
        owned_objects: Sequence[MemoryObj],
    ) -> None:
        """Retain a source while the retriever still owns its allocations.

        The interval must be covered by whole, consecutive cache objects on
        every layer. Unmatched requests or incomplete coverage remain absent.
        The caller must still hold the retrieval references in owned_objects.
        """
        with self._lock:
            if self._state != "staging":
                raise RuntimeError("Selected prefill step is not staging")
            request = self._requests.get(request_id)
            if request is None or not layers or len(starts) != len(ends):
                return
            if request_id in self._sources:
                raise RuntimeError("Selected prefill source staged twice")
            first = next(
                (i for i, start in enumerate(starts) if start == request.start),
                None,
            )
            if first is None:
                return
            stop = first
            cursor = request.start
            spans: list[tuple[int, int]] = []
            while stop < len(starts) and cursor < request.end:
                start, end = starts[stop], ends[stop]
                if start != cursor or not start < end <= request.end:
                    return
                spans.append((start, end))
                cursor = end
                stop += 1
            if cursor != request.end:
                return

            selected: list[LayerPageSource | tuple[MemoryObj, ...]] = []
            owned_ids = {id(obj) for obj in owned_objects}
            new_owners: dict[int, TensorMemoryObj] = {}
            for layer_id, layer in enumerate(layers):
                if layer is None:
                    return
                if isinstance(layer, LayerPageSource):
                    if layer.layer_id != layer_id:
                        return
                    objects = layer.pages + layer.suffix
                    page_count = len(layer.pages)
                    suffix_start = max(0, first - page_count)
                    suffix_stop = max(0, stop - page_count)
                    selected.append(
                        LayerPageSource(
                            layer.pages[first:stop],
                            layer_id,
                            layer.suffix[suffix_start:suffix_stop],
                        )
                    )
                else:
                    objects = layer
                    selected.append(tuple(objects[first:stop]))
                if len(objects) != len(starts):
                    return
                for obj, (start, end) in zip(
                    objects[first:stop], spans, strict=True
                ):
                    if (
                        id(obj) not in owned_ids
                        or not isinstance(obj, TensorMemoryObj)
                        or not obj.is_valid()
                        or obj.get_num_tokens() < end - start
                    ):
                        return
                    if id(obj) not in self._owners:
                        new_owners[id(obj)] = obj

            retained: list[TensorMemoryObj] = []
            try:
                for obj in new_owners.values():
                    obj.ref_count_up()
                    retained.append(obj)
            except BaseException:
                for obj in reversed(retained):
                    obj.ref_count_down()
                raise
            self._owners.update(new_owners)
            self._sources[request_id] = SelectedPrefillSource(
                request, tuple(spans), tuple(selected)
            )

    def seal(self) -> None:
        """Publish the sources prepared during successful connector setup."""
        with self._lock:
            if self._state != "staging":
                raise RuntimeError("Selected prefill step is not staging")
            self._state = "ready"

    def acquire(self) -> tuple[SelectedPrefillSource, ...]:
        """Hand the prepared sources to the one consumer of this step."""
        with self._lock:
            if self._state != "ready":
                raise RuntimeError("Selected prefill sources are not available")
            self._state = "acquired"
            return tuple(self._sources.values())

    def release(self, completion: PrefillReadCompletion | None = None) -> None:
        """Retire the step once its selected reader has completed.

        After acquiring a non-empty source set, provide an event even if forward
        aborted. If synchronization raises, ownership stays live and release
        can be retried. The caller must stop submitting reads before release.
        """
        with self._lock:
            if self._state == "released":
                return
            if self._state == "acquired" and self._sources:
                if completion is None:
                    raise ValueError("Selected prefill reader completion is required")
                completion.synchronize()
            self._sources.clear()
            while self._owners:
                _, obj = self._owners.popitem()
                obj.ref_count_down()
            self._state = "released"
