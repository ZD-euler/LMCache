# SPDX-License-Identifier: Apache-2.0

# Standard
from collections.abc import Generator
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1 import cache_engine as cache_engine_module
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.memory_management import (
    LayerPageMemoryObj,
    LayerPageSource,
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.selected_prefill import SelectedPrefillRequest, SelectedPrefillStep


def make_object(
    tokens: int, allocator: MemoryAllocatorInterface | None = None
) -> TensorMemoryObj:
    data = torch.zeros(tokens, dtype=torch.uint8)
    metadata = MemoryObjMetadata(
        shape=data.shape,
        dtype=data.dtype,
        address=0,
        phy_size=tokens,
        ref_count=1,
        fmt=MemoryFormat.BINARY_BUFFER,
    )
    return TensorMemoryObj(data, metadata, parent_allocator=allocator)


class Completion:
    def __init__(self, owners: tuple[TensorMemoryObj, ...]) -> None:
        self.owners = owners
        self.fail = False
        self.calls = 0

    def synchronize(self) -> None:
        self.calls += 1
        assert all(obj.get_ref_count() == 1 for obj in self.owners)
        if self.fail:
            raise RuntimeError("stream failed")


def test_selected_prefill_keeps_shared_page_and_partial_tail_live() -> None:
    page = LayerPageMemoryObj(
        torch.zeros(8, dtype=torch.uint8),
        MemoryObjMetadata(
            shape=torch.Size([4]),
            dtype=torch.uint8,
            address=0,
            phy_size=8,
            ref_count=1,
            fmt=MemoryFormat.BINARY_BUFFER,
        ),
        parent_allocator=None,
        num_layers=2,
        group_prefix_sum=(0, 4, 8),
    )
    tail0, tail1 = make_object(2), make_object(2)
    owners = (page, tail0, tail1)
    layers = (
        LayerPageSource((page,), 0, (tail0,)),
        LayerPageSource((page,), 1, (tail1,)),
    )
    step = SelectedPrefillStep(
        (
            SelectedPrefillRequest("r1", 4, 10),
            SelectedPrefillRequest("r2", 4, 10),
        )
    )
    step.begin()
    for request_id in ("r1", "r2"):
        step.stage(request_id, (4, 8), (8, 10), layers, owners)
    assert [obj.get_ref_count() for obj in owners] == [2, 2, 2]
    step.seal()
    sources = step.acquire()
    assert [source.request.request_id for source in sources] == ["r1", "r2"]
    assert all(source.spans == ((4, 8), (8, 10)) for source in sources)
    assert all(source.layers == layers for source in sources)
    for obj in owners:
        obj.ref_count_down()

    with pytest.raises(ValueError, match="completion is required"):
        step.release()
    event = Completion(owners)
    event.fail = True
    with pytest.raises(RuntimeError, match="stream failed"):
        step.release(event)
    assert [obj.get_ref_count() for obj in owners] == [1, 1, 1]
    event.fail = False
    step.release(event)
    step.release()
    assert event.calls == 2
    assert [obj.get_ref_count() for obj in owners] == [0, 0, 0]


@pytest.mark.parametrize(
    ("start", "end", "starts", "ends", "owned"),
    [
        (1, 8, (0, 4), (4, 8), True),
        (0, 7, (0, 4), (4, 8), True),
        (0, 8, (0, 5), (4, 8), True),
        (0, 8, (0, 4), (4, 8), False),
    ],
)
def test_selected_prefill_requires_exact_owned_coverage(
    start: int,
    end: int,
    starts: tuple[int, ...],
    ends: tuple[int, ...],
    owned: bool,
) -> None:
    first, second = make_object(4), make_object(4)
    step = SelectedPrefillStep((SelectedPrefillRequest("r", start, end),))
    step.begin()
    step.stage(
        "r",
        starts,
        ends,
        ((first, second),),
        (first, second) if owned else (second,),
    )
    step.seal()
    assert step.acquire() == ()
    step.release()
    assert first.get_ref_count() == second.get_ref_count() == 1


def test_selected_prefill_setup_failure_releases_unpublished_sources() -> None:
    owner = make_object(4)
    step = SelectedPrefillStep((SelectedPrefillRequest("r", 0, 4),))
    step.begin()
    step.stage("r", (0,), (4,), ((owner,),), (owner,))
    assert owner.get_ref_count() == 2
    with pytest.raises(RuntimeError, match="not available"):
        step.acquire()
    step.release()
    assert owner.get_ref_count() == 1
    with pytest.raises(RuntimeError, match="already used"):
        step.begin()


class LocalStorage:
    def __init__(self, layers: list[list[TensorMemoryObj]]) -> None:
        self.layers = layers

    def contains(self, key: CacheEngineKey, locations: list[str]) -> str:
        return "LocalCPUBackend"

    def layerwise_batched_get(
        self, keys: list[list[CacheEngineKey]], location: str
    ) -> Generator[Future[list[TensorMemoryObj]], None, None]:
        assert location == "LocalCPUBackend"
        for layer_keys, owners in zip(keys, self.layers, strict=True):
            assert len(layer_keys) == len(owners)
            future: Future[list[TensorMemoryObj]] = Future()
            future.set_result(owners)
            yield future


class NativeLoadRecorder:
    def __init__(self) -> None:
        self.loaded: list[list[TensorMemoryObj]] = []

    def batched_to_gpu(
        self, starts: list[int], ends: list[int], **kwargs: Any
    ) -> Generator[None, list[TensorMemoryObj], None]:
        assert (starts, ends) == ([0, 4], [4, 6])
        assert "prefill_c8_memory_objs" not in kwargs
        assert "selected_prefill_step" not in kwargs
        while True:
            objects = yield
            if objects is not None:
                self.loaded.append(objects)


@pytest.mark.parametrize("capture", [False, True])
def test_single_rank_deferred_retrieve_submits_every_native_layer(
    monkeypatch: pytest.MonkeyPatch, capture: bool
) -> None:
    # Fake storage and device calls only. Exercise the real public retriever,
    # including its world-size routing and deferred generator lifecycle.
    allocator = Mock(spec=MemoryAllocatorInterface)
    layers = [
        [make_object(4, allocator), make_object(2, allocator)] for _ in range(2)
    ]
    owners = tuple(obj for layer in layers for obj in layer)
    connector = NativeLoadRecorder()
    engine = LMCacheEngine.__new__(LMCacheEngine)
    engine._init_failed = False
    engine._health_monitor = None
    engine.enable_shared_cpu_cache = True
    engine.use_layerwise = True
    engine.save_only_first_rank = True
    engine.metadata = SimpleNamespace(
        world_size=1, use_mla=True, is_first_rank=lambda: True
    )
    engine.dsa_two_groups = True
    engine._transfer_layer_counts = (2, 2)
    engine.retrieve_locations = ["LocalCPUBackend"]
    engine.storage_manager = LocalStorage(layers)
    engine.gpu_connector = connector
    engine.stats_monitor = SimpleNamespace(
        on_retrieve_request=lambda _: 1, on_retrieve_finished=lambda *_: None
    )
    keys = [CacheEngineKey("test", 1, 0, i, torch.float16) for i in range(2)]
    engine.token_database = SimpleNamespace(
        process_tokens=lambda **_: iter([(0, 4, keys[0]), (4, 6, keys[1])])
    )
    monkeypatch.setattr(
        cache_engine_module, "assert_layerwise_gpu_connector", lambda _: None
    )
    step = None
    kwargs = {}
    if capture:
        step = SelectedPrefillStep((SelectedPrefillRequest("r", 0, 6),))
        step.begin()
        kwargs["selected_prefill_step"] = step

    retriever = engine.retrieve_layer(
        [1, 2, 3, 4, 5, 6], req_id="r", deferred_layerwise_get=True, **kwargs
    )
    assert int(next(retriever)) == 6
    assert not connector.loaded
    assert all(obj.get_ref_count() == (2 if capture else 1) for obj in owners)
    assert next(retriever) is None
    assert connector.loaded == layers[:1]
    assert next(retriever) is None
    assert connector.loaded == layers
    if step is not None:
        step.seal()
        (source,) = step.acquire()
        assert source.spans == ((0, 4), (4, 6))
        assert source.layers == tuple(tuple(layer) for layer in layers)

    assert next(retriever).tolist() == [True] * 6
    with pytest.raises(StopIteration):
        next(retriever)

    if step is not None:
        allocator.free.assert_not_called()
        event = Completion(owners)
        event.fail = True
        with pytest.raises(RuntimeError, match="stream failed"):
            step.release(event)
        allocator.free.assert_not_called()
        assert all(obj.get_ref_count() == 1 for obj in owners)
        event.fail = False
        step.release(event)
        step.release(event)
        assert event.calls == 2

    assert allocator.free.call_count == len(owners)
    for obj in owners:
        assert sum(call.args == (obj,) for call in allocator.free.call_args_list) == 1
        assert obj.get_ref_count() == 0
