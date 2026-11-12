# SPDX-License-Identifier: Apache-2.0
"""Local ownership and completion contract for shared prefill source reads."""

# Standard
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum, auto
from threading import Lock

# First Party
from lmcache.v1.memory_management import TensorMemoryObj
from lmcache.v1.selected_prefill import PrefillReadCompletion
from lmcache.v1.shared_cpu_cache import PassiveSharedViewAllocator


@dataclass(frozen=True, slots=True)
class PrefillReadDomain:
    """One TP group, KV group, and generation of a physical shared slab."""

    tp_group: str
    kv_group: int
    slab_name: str
    generation: int

    def __post_init__(self) -> None:
        if not self.tp_group or not self.slab_name:
            raise ValueError("Prefill lease requires a TP group and slab")
        if self.kv_group < 0 or self.generation < 0:
            raise ValueError("Invalid prefill KV group or slab generation")


@dataclass(frozen=True, slots=True)
class PrefillReadStep:
    """Identity of a prefill source publication within a read domain."""

    domain: PrefillReadDomain
    sequence: int


class PrefillReadResult(Enum):
    """Evidence that a rank has no remaining selected readers."""

    NO_READ = auto()
    COMPLETED = auto()


@dataclass(frozen=True, slots=True)
class PrefillReadAck:
    """Completion evidence to be carried by the ordered TP transport."""

    step: PrefillReadStep
    rank: int
    result: PrefillReadResult


class PrefillRankReader:
    """Track one rank's submissions for one step.

    The connector must use one authoritative reader per rank and step, and
    route every selected read through that reader.
    """

    def __init__(self, step: PrefillReadStep, rank: int) -> None:
        self._step = step
        self._rank = rank
        self._lock = Lock()
        self._submitted = False
        self._sealed = False
        self._completion: PrefillReadCompletion | None = None
        self._ack: PrefillReadAck | None = None

    def submit_read(self, enqueue: Callable[[], None]) -> None:
        """Enqueue a selected read and track even a partial submission failure."""
        with self._lock:
            if self._sealed:
                raise RuntimeError("Prefill reader is sealed")
            self._submitted = True
            enqueue()

    def finish(
        self,
        record_completion: Callable[[], PrefillReadCompletion] | None = None,
    ) -> PrefillReadAck:
        """Seal the reader and acknowledge after its selected reads are idle.

        Record an event that covers all submitted work on this rank, including
        work queued before an exception. An event is required after the first
        submission attempt. A failed synchronization can retry the same event.
        """
        with self._lock:
            self._sealed = True
            if self._ack is not None:
                return self._ack
            result = PrefillReadResult.NO_READ
            if self._submitted:
                if self._completion is None:
                    if record_completion is None:
                        raise ValueError("Submitted prefill reads require completion")
                    self._completion = record_completion()
                self._completion.synchronize()
                result = PrefillReadResult.COMPLETED
            self._ack = PrefillReadAck(self._step, self._rank, result)
            return self._ack


@dataclass(slots=True)
class _ReadLease:
    owners: list[TensorMemoryObj]
    allocation_spans: dict[int, tuple[int, int]]
    acks: dict[int, PrefillReadResult] = field(default_factory=dict)
    quarantined: bool = False


class Rank0PrefillReadLedger:
    """Own canonical slab allocations until every expected reader is idle.

    Keep this ledger alive for the rank0 worker's lifetime. The caller must
    retain its real allocator references throughout ``begin`` and publish
    only after it succeeds. The ledger takes references independent of
    request leases and temporary pins. Quarantined steps keep their owners.

    The connector must bind the domain to the actual slab mapping and assign
    one authoritative reader to each expected rank. Before publishing, it
    verifies that all ranks enrolled the same step. Acknowledgements must
    arrive through an ordered transport that identifies their actual senders.
    If completion is unknown, the connector must stop further publication,
    quarantine the affected step, and keep this ledger alive. This class
    does not drive transport, deadlines, or worker shutdown.
    """

    def __init__(self, domain: PrefillReadDomain, ranks: Iterable[int]) -> None:
        self._domain = domain
        self._ranks = frozenset(ranks)
        if not self._ranks or min(self._ranks) < 0:
            raise ValueError("Invalid prefill reader ranks")
        self._sequence = -1
        self._leases: dict[PrefillReadStep, _ReadLease] = {}
        self._lock = Lock()

    def begin(
        self, sequence: int, owners: Iterable[TensorMemoryObj]
    ) -> PrefillReadStep:
        """Acquire one reference per live, canonical rank0 allocation."""
        with self._lock:
            if sequence <= self._sequence:
                raise ValueError("Prefill read sequence must increase")
            unique = {id(obj): obj for obj in owners}
            if not unique:
                raise ValueError("Prefill read step requires source allocations")
            active_spans = {
                identity: span
                for lease in self._leases.values()
                for identity, span in lease.allocation_spans.items()
            }
            new_spans: dict[int, tuple[int, int]] = {}
            for identity, obj in unique.items():
                if (
                    not isinstance(obj, TensorMemoryObj)
                    or obj.parent_allocator is None
                    or isinstance(obj.parent_allocator, PassiveSharedViewAllocator)
                    or not obj.is_valid()
                    or obj.get_ref_count() <= 0
                ):
                    raise ValueError("Prefill source lacks a live allocator owner")
                start, size = obj.data_ptr, obj.get_physical_size()
                if start <= 0 or size <= 0:
                    raise ValueError("Prefill source has an invalid allocation range")
                span = (start, start + size)
                if active_spans.get(identity, span) != span:
                    raise ValueError("Live prefill source allocation changed")
                new_spans[identity] = span
            spans = sorted((active_spans | new_spans).values())
            if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
                raise ValueError("Distinct prefill source owners overlap")
            retained: list[TensorMemoryObj] = []
            try:
                for obj in unique.values():
                    obj.ref_count_up()
                    retained.append(obj)
            except BaseException:
                # No address has been published yet.
                for obj in reversed(retained):
                    obj.ref_count_down()
                raise
            step = PrefillReadStep(self._domain, sequence)
            self._leases[step] = _ReadLease(retained, new_spans)
            self._sequence = sequence
            return step

    def acknowledge(self, ack: PrefillReadAck, *, sender_rank: int) -> None:
        """Record a completion acknowledgement from the transport's sender."""
        with self._lock:
            lease = self._leases.get(ack.step)
            if lease is None or lease.quarantined:
                raise ValueError("Unknown or quarantined prefill read step")
            if sender_rank != ack.rank or sender_rank not in self._ranks:
                raise ValueError("Unexpected prefill acknowledgement sender")
            if not isinstance(ack.result, PrefillReadResult):
                raise ValueError("Invalid prefill acknowledgement result")
            previous = lease.acks.get(sender_rank)
            if previous is not None and previous != ack.result:
                lease.quarantined = True
                raise ValueError("Conflicting prefill acknowledgement")
            lease.acks[sender_rank] = ack.result

    def quarantine(self, step: PrefillReadStep) -> None:
        """Hold a step after transport failure or unprovable reader state."""
        with self._lock:
            self._leases[step].quarantined = True

    def retire(self, step: PrefillReadStep) -> bool:
        """Release once every expected rank has acknowledged the exact step."""
        with self._lock:
            lease = self._leases[step]
            if lease.quarantined or lease.acks.keys() != self._ranks:
                return False
            try:
                while lease.owners:
                    owner = lease.owners[-1]
                    owner.ref_count_down()
                    lease.owners.pop()
            except BaseException:
                # Refcount may already be decremented. Keep the failed owner
                # reachable and forbid another release attempt for this step.
                lease.quarantined = True
                raise
            del self._leases[step]
            return True
