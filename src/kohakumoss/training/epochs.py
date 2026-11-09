"""Translate the dataset's actual batch plans into optimiser-step boundaries."""


def epoch_steps(
    planner, epochs: int, data_world: int, accumulation: int
) -> tuple[int, ...]:
    batches_per_step = data_world * accumulation
    if data_world > 1 and planner.balance > 1 and planner.balance != accumulation:
        raise ValueError(
            "epoch mode requires data balancing to match gradient accumulation"
        )
    total = 0
    boundaries = []
    for epoch in range(epochs):
        length = planner.epoch_length(epoch)
        if length <= 0:
            raise ValueError(f"planner produced an empty epoch {epoch + 1}")
        total += length
        if total % batches_per_step:
            raise ValueError(
                f"epoch {epoch + 1} ends after {total} batches, which cannot align with "
                f"{data_world} data ranks and {accumulation} batches per optimiser step; "
                "adjust batch size, data ranks or gradient accumulation"
            )
        boundaries.append(total // batches_per_step)
    return tuple(boundaries)
