# SPDX-License-Identifier: Apache-2.0

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event, Lock
from unittest.mock import Mock
from weakref import ref
import gc

# Third Party
import pytest
import torch

# First Party
from lmcache.v1 import prefill_read_lease as lease_module
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.prefill_read_lease import (
    PrefillRankReader,
    PrefillReadAck,
    PrefillReadDomain,
    PrefillReadResult,
    Rank0PrefillReadLedger,
)
from lmcache.v1.shared_cpu_cache import PassiveSharedViewAllocator


def make_owner(
    allocator: MemoryAllocatorInterface, data: torch.Tensor | None = None
) -> TensorMemoryObj:
    if data is None:
        data = torch.zeros(4, dtype=torch.uint8)
    return TensorMemoryObj(
        data,
        MemoryObjMetadata(
            shape=data.shape,
            dtype=data.dtype,
            address=0,
            phy_size=data.numel(),
            ref_count=1,
            fmt=MemoryFormat.BINARY_BUFFER,
        ),
        parent_allocator=allocator,
    )


@pytest.fixture
def allocator() -> Mock:
    allocator = Mock(spec=MemoryAllocatorInterface)
    allocator.free.side_effect = lambda obj: obj.invalidate()
    return allocator


@pytest.fixture
def domain() -> PrefillReadDomain:
    return PrefillReadDomain("tp-0", 0, "/test-slab", 7)


def test_rank0_retires_only_after_every_rank_is_idle(
    domain: PrefillReadDomain, allocator: Mock
) -> None:
    first, second = make_owner(allocator), make_owner(allocator)
    ledger = Rank0PrefillReadLedger(domain, (0, 1, 2))
    step = ledger.begin(3, (first, first, second))
    assert [obj.get_ref_count() for obj in (first, second)] == [2, 2]
    # Request owners can disappear independently of the readers.
    for obj in (first, second):
        obj.ref_count_down()
    for rank in (0, 1):
        ack = PrefillRankReader(step, rank).finish()
        assert ack.result is PrefillReadResult.NO_READ
        ledger.acknowledge(ack, sender_rank=rank)
    ledger.acknowledge(ack, sender_rank=1)
    assert not ledger.retire(step)
    allocator.free.assert_not_called()

    remote = PrefillRankReader(step, 2)
    remote.submit_read(lambda: None)
    completion = Mock()
    completion.synchronize.side_effect = RuntimeError("device event failed")
    with pytest.raises(RuntimeError, match="event failed"):
        remote.finish(lambda: completion)
    assert not ledger.retire(step)
    allocator.free.assert_not_called()
    with pytest.raises(RuntimeError, match="sealed"):
        remote.submit_read(lambda: None)

    completion.synchronize.side_effect = None
    ack = remote.finish()
    assert ack.result is PrefillReadResult.COMPLETED
    assert completion.synchronize.call_count == 2
    ledger.acknowledge(ack, sender_rank=2)
    assert ledger.retire(step)
    assert allocator.free.call_count == 2
    allocator.free.assert_any_call(first)
    allocator.free.assert_any_call(second)
    with pytest.raises(ValueError, match="sequence must increase"):
        ledger.begin(3, (first,))


def test_failed_enqueue_requires_completion_before_ack(
    domain: PrefillReadDomain, allocator: Mock
) -> None:
    owner = make_owner(allocator)
    ledger = Rank0PrefillReadLedger(domain, (0, 1))
    step = ledger.begin(0, (owner,))
    owner.ref_count_down()
    reader = PrefillRankReader(step, 1)
    enqueue = Mock(side_effect=RuntimeError("partial enqueue"))
    with pytest.raises(RuntimeError, match="partial enqueue"):
        reader.submit_read(enqueue)
    with pytest.raises(ValueError, match="require completion"):
        reader.finish()
    allocator.free.assert_not_called()

    completion = Mock()
    ledger.acknowledge(reader.finish(lambda: completion), sender_rank=1)
    completion.synchronize.assert_called_once_with()
    ledger.acknowledge(PrefillRankReader(step, 0).finish(), sender_rank=0)
    assert ledger.retire(step)
    allocator.free.assert_called_once_with(owner)


@pytest.mark.parametrize(
    "other_domain",
    [
        PrefillReadDomain("tp-1", 0, "/test-slab", 7),
        PrefillReadDomain("tp-0", 1, "/test-slab", 7),
        PrefillReadDomain("tp-0", 0, "/other-slab", 7),
        PrefillReadDomain("tp-0", 0, "/test-slab", 8),
    ],
)
def test_ack_requires_exact_domain_and_sender(
    domain: PrefillReadDomain,
    other_domain: PrefillReadDomain,
    allocator: Mock,
) -> None:
    owner = make_owner(allocator)
    ledger = Rank0PrefillReadLedger(domain, (0, 1))
    step = ledger.begin(1, (owner,))
    owner.ref_count_down()
    ack = PrefillRankReader(step, 0).finish()
    stale_step = replace(step, domain=other_domain)
    for wrong_step in (stale_step, replace(step, sequence=0)):
        with pytest.raises(ValueError, match="Unknown or quarantined"):
            ledger.acknowledge(replace(ack, step=wrong_step), sender_rank=0)
    with pytest.raises(ValueError, match="Unexpected"):
        ledger.acknowledge(ack, sender_rank=1)
    with pytest.raises(ValueError, match="Unexpected"):
        ledger.acknowledge(replace(ack, rank=2), sender_rank=2)
    ledger.acknowledge(PrefillRankReader(step, 1).finish(), sender_rank=1)
    assert not ledger.retire(step)
    allocator.free.assert_not_called()
    ledger.acknowledge(ack, sender_rank=0)
    assert ledger.retire(step)
    allocator.free.assert_called_once_with(owner)


@pytest.mark.parametrize("conflict", [False, True])
def test_unknown_or_conflicting_reader_state_preserves_owner(
    domain: PrefillReadDomain, allocator: Mock, conflict: bool
) -> None:
    owner = make_owner(allocator)
    ledger = Rank0PrefillReadLedger(domain, (0, 1))
    step = ledger.begin(0, (owner,))
    owner.ref_count_down()
    ledger.acknowledge(PrefillRankReader(step, 0).finish(), sender_rank=0)
    if conflict:
        with pytest.raises(ValueError, match="Conflicting"):
            ledger.acknowledge(
                PrefillReadAck(step, 0, PrefillReadResult.COMPLETED), sender_rank=0
            )
    else:
        ledger.quarantine(step)
    with pytest.raises(ValueError, match="quarantined"):
        ledger.acknowledge(PrefillRankReader(step, 1).finish(), sender_rank=1)
    assert not ledger.retire(step)
    assert owner.get_ref_count() == 1
    allocator.free.assert_not_called()
    # In this test no device work exists and worker teardown is safe.
    owner.invalidate()


def test_ledger_rejects_aliases_and_passive_views(
    domain: PrefillReadDomain, allocator: Mock
) -> None:
    ledger = Rank0PrefillReadLedger(domain, (0, 1))
    buffer = torch.zeros(8, dtype=torch.uint8)
    first = make_owner(allocator, buffer[:4])
    alias = make_owner(allocator, buffer[2:6])
    with pytest.raises(ValueError, match="overlap"):
        ledger.begin(0, (first, alias))
    assert first.get_ref_count() == alias.get_ref_count() == 1

    passive = make_owner(Mock(spec=PassiveSharedViewAllocator))
    with pytest.raises(ValueError, match="allocator owner"):
        ledger.begin(0, (passive,))
    assert passive.get_ref_count() == 1
    for obj in (first, alias, passive):
        obj.invalidate()


def test_overlapping_steps_hold_separate_references(
    domain: PrefillReadDomain, allocator: Mock
) -> None:
    buffer = torch.zeros(8, dtype=torch.uint8)
    owner = make_owner(allocator, buffer[:4])
    ledger = Rank0PrefillReadLedger(domain, (0, 1))
    first = ledger.begin(0, (owner,))
    alias = make_owner(allocator, buffer[2:6])
    with pytest.raises(ValueError, match="overlap"):
        ledger.begin(1, (alias,))
    alias.invalidate()
    second = ledger.begin(1, (owner,))
    owner.ref_count_down()

    for rank in (0, 1):
        ledger.acknowledge(PrefillRankReader(first, rank).finish(), sender_rank=rank)
    assert ledger.retire(first)
    assert owner.get_ref_count() == 1
    allocator.free.assert_not_called()

    for rank in (0, 1):
        ledger.acknowledge(PrefillRankReader(second, rank).finish(), sender_rank=rank)
    assert ledger.retire(second)
    allocator.free.assert_called_once_with(owner)


class ObservedLock:
    """Report a failed nonblocking acquisition before waiting on a real lock."""

    def __init__(self) -> None:
        self.blocked = Event()
        self._lock = Lock()

    def __enter__(self) -> "ObservedLock":
        if not self._lock.acquire(blocking=False):
            self.blocked.set()
            if not self._lock.acquire(timeout=10):
                raise TimeoutError("Contender could not acquire the test lock")
        return self

    def __exit__(self, *_: object) -> None:
        self._lock.release()


def test_reader_waits_for_concurrent_enqueue_before_recording_completion(
    domain: PrefillReadDomain, allocator: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = make_owner(allocator)
    ledger = Rank0PrefillReadLedger(domain, (0,))
    step = ledger.begin(0, (owner,))
    owner.ref_count_down()
    reader_lock = ObservedLock()
    with monkeypatch.context() as scoped:
        scoped.setattr(lease_module, "Lock", lambda: reader_lock)
        reader = PrefillRankReader(step, 0)
    enqueue_entered = Event()
    enqueue_allowed_to_finish = Event()
    enqueue_finished = Event()
    completion = Mock()

    def enqueue() -> None:
        enqueue_entered.set()
        assert enqueue_allowed_to_finish.wait(5)
        enqueue_finished.set()

    def record_completion() -> Mock:
        assert enqueue_finished.is_set()
        return completion

    with ThreadPoolExecutor(max_workers=2) as executor:
        submit = executor.submit(reader.submit_read, enqueue)
        try:
            assert enqueue_entered.wait(5)
            finish_result = executor.submit(reader.finish, record_completion)
            # Enqueue still owns the lock when finish attempts to acquire it.
            assert reader_lock.blocked.wait(5)
            assert not finish_result.done()
            completion.synchronize.assert_not_called()
        finally:
            enqueue_allowed_to_finish.set()
        submit.result(timeout=5)
        ack = finish_result.result(timeout=5)

    assert ack.result is PrefillReadResult.COMPLETED
    completion.synchronize.assert_called_once_with()
    with pytest.raises(RuntimeError, match="sealed"):
        reader.submit_read(lambda: None)
    ledger.acknowledge(ack, sender_rank=0)
    assert ledger.retire(step)
    allocator.free.assert_called_once_with(owner)


def test_concurrent_retire_frees_once(
    domain: PrefillReadDomain, allocator: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = make_owner(allocator)
    ledger_lock = ObservedLock()
    with monkeypatch.context() as scoped:
        scoped.setattr(lease_module, "Lock", lambda: ledger_lock)
        ledger = Rank0PrefillReadLedger(domain, (0,))
    step = ledger.begin(0, (owner,))
    owner.ref_count_down()
    ledger.acknowledge(PrefillRankReader(step, 0).finish(), sender_rank=0)
    free_entered = Event()
    free_allowed_to_finish = Event()

    def free(obj: TensorMemoryObj) -> None:
        free_entered.set()
        assert free_allowed_to_finish.wait(5)
        obj.invalidate()

    allocator.free.side_effect = free
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(ledger.retire, step)
        try:
            assert free_entered.wait(5)
            second = executor.submit(ledger.retire, step)
            # The first retire still owns the lock inside allocator.free().
            assert ledger_lock.blocked.wait(5)
            assert not second.done()
        finally:
            free_allowed_to_finish.set()
        assert first.result(timeout=5)
        with pytest.raises(KeyError):
            second.result(timeout=5)
    allocator.free.assert_called_once_with(owner)


def test_begin_rolls_back_retained_owners_and_can_retry_sequence(
    domain: PrefillReadDomain, allocator: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = make_owner(allocator), make_owner(allocator)
    ledger = Rank0PrefillReadLedger(domain, (0, 1))

    def fail_retain() -> None:
        assert first.get_ref_count() == 2
        raise RuntimeError("retain failed")

    with monkeypatch.context() as scoped:
        scoped.setattr(second, "ref_count_up", fail_retain)
        with pytest.raises(RuntimeError, match="retain failed"):
            ledger.begin(4, (first, second))
    assert [obj.get_ref_count() for obj in (first, second)] == [1, 1]
    allocator.free.assert_not_called()

    step = ledger.begin(4, (first, second))
    assert [obj.get_ref_count() for obj in (first, second)] == [2, 2]
    first.ref_count_down()
    second.ref_count_down()
    for rank in (0, 1):
        ledger.acknowledge(PrefillRankReader(step, rank).finish(), sender_rank=rank)
    assert ledger.retire(step)
    assert allocator.free.call_count == 2
    allocator.free.assert_any_call(first)
    allocator.free.assert_any_call(second)


def test_failed_free_keeps_owner_quarantined_without_retry(
    domain: PrefillReadDomain, allocator: Mock
) -> None:
    other, owner = make_owner(allocator), make_owner(allocator)
    ledger = Rank0PrefillReadLedger(domain, (0,))
    step = ledger.begin(0, (other, owner))
    other.ref_count_down()
    owner.ref_count_down()
    ledger.acknowledge(PrefillRankReader(step, 0).finish(), sender_rank=0)

    def fail_free(obj: TensorMemoryObj) -> None:
        assert obj is owner
        raise RuntimeError("allocator free failed")

    allocator.free.side_effect = fail_free
    with pytest.raises(RuntimeError, match="allocator free failed"):
        ledger.retire(step)
    assert owner.get_ref_count() == 0
    assert other.get_ref_count() == 1
    assert not ledger.retire(step)
    allocator.free.assert_called_once_with(owner)

    # Clear the mock's references so only the ledger can keep the owner live.
    owner_ref = ref(owner)
    allocator.reset_mock(side_effect=True)
    del owner
    gc.collect()
    failed_owner = owner_ref()
    assert failed_owner is not None
    allocator.free.assert_not_called()
    # Simulate safe worker teardown after this host-only failure injection.
    failed_owner.invalidate()
    other.invalidate()
