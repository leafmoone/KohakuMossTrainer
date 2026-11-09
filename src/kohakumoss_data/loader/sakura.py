"""Step-controlled, single-consumer TAR stream from Sakuradatasetservice."""

import glob
from dataclasses import replace
from pathlib import Path

from kohakumoss_common.log import get_logger
from kohakumoss_data.dataset import MossDataset
from kohakumoss_data.format.catalog import Catalog
from kohakumoss_data.loader.stateless import resolve_transform
from kohakumoss_data.registry import SOURCES
from kohakumoss_data.seeds import seed_of
from kohakumoss_data.sources.webdataset import WebDatasetSource

log = get_logger(__name__)


@SOURCES.register("sakura_tar")
class SakuraTarSource(WebDatasetSource):
    """Scan a leased TAR in memory, without writing into the service cache."""

    def catalog(self):
        return Catalog(list(self.scan()))


class SakuraMossDataset(MossDataset):
    """Use the configured bound for prewarm: later shards are not known yet."""

    def text_tokens_max(self):
        return self.cfg.max_text_tokens


class SakuraLoader:
    """Hold each shard lease while decoding its batches, then acknowledge it.

    Checkpoints restore model state but do not rewind the external service queue.
    A resumed run can repeat the current shard or skip previously acknowledged work.
    """

    def __init__(self, cfg, transform=None):
        from sakuradatasetservice import DataServiceClient

        entry = cfg.datasets[0]
        socket = Path(entry["socket"]).expanduser().resolve()
        self.client = DataServiceClient(
            socket, worker_count=1, request_timeout_seconds=30.0
        )
        self.leases = iter(self.client.iter_leases(worker_id=0))
        self.cfg = cfg
        self.transform = resolve_transform(transform)
        self.start = 0
        self._lease = None
        self.dataset = self._next_dataset()

    def _next_dataset(self):
        if self._lease is not None:
            self._close_handles()
            self._lease.acknowledge()
        try:
            self._lease = next(self.leases)
        except StopIteration:
            raise RuntimeError(
                "Sakura service exhausted: start it with --repeat"
            ) from None
        descriptor = self._lease.descriptor
        log.info(
            f"Sakura allocation round {descriptor.cycle_index + 1}: "
            f"{descriptor.record.path}"
        )
        old = self.cfg.datasets[0]
        entry = {
            key: old[key] for key in ("weight", "caption", "loss_weight") if key in old
        }
        entry.update(
            source="sakura_tar",
            path=glob.escape(str(descriptor.local_path)),
            txt_as=old.get("txt_as", "tags"),
        )
        seed = seed_of(
            "sakura", self.cfg.seed, descriptor.cycle_index, descriptor.record.path
        )
        cfg = replace(
            self.cfg,
            datasets=[entry],
            data_scan_workers=0,
            seed=seed,
            planner_kwargs={**self.cfg.planner_kwargs, "drop_last": False},
        )
        dataset = SakuraMossDataset(cfg, writer=False)
        if not dataset.planner.epoch_length(0):
            raise ValueError(
                f"Sakura shard has no trainable batches: {descriptor.local_path}"
            )
        return dataset

    def _close_handles(self):
        for source in self.dataset.sources:
            for handle in getattr(source, "_handles", {}).values():
                handle.close()
            if hasattr(source, "_handles"):
                source._handles.clear()

    def __iter__(self):
        dataset = self.dataset
        while True:
            for index in range(dataset.planner.epoch_length(0)):
                batch = self.transform(dataset.materialize(index, 0, 1))
                self.start += 1
                yield batch
            dataset = self._next_dataset()
            self.dataset = dataset

    def state_dict(self):
        return {"seed": self.cfg.seed, "start": self.start}

    def load_state_dict(self, state):
        if state["seed"] != self.cfg.seed:
            raise ValueError("checkpoint data seed differs from Sakura stream seed")
        # The service's durable queue is authoritative, not the checkpoint cursor.
        self.start = int(state["start"])
        log.warning("Sakura service queue is not rewound on checkpoint resume")
