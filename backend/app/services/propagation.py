import math


def propagate_delay(
    current_delay,
    predicted_changes,
):
    """
    Dynamically propagate predicted delay changes.

    The first prediction is used directly.
    Longer-horizon predictions are progressively damped
    because recursive model forecasts accumulate uncertainty.
    """

    results = []

    delay = float(current_delay)

    for step, raw_change in enumerate(
        predicted_changes,
        start=1,
    ):

        raw_change = float(raw_change)

        # Progressive damping.
        #
        # Horizon 1 -> 100%
        # Horizon 2 -> 70%
        # Horizon 3 -> 55%
        # Horizon 4 -> 45%
        #
        # This reduces recursive error amplification while
        # preserving the model's learned direction.

        damping = 1.0 / math.sqrt(step)

        change = raw_change * damping

        delay += change

        # Uncertainty grows with forecast horizon.
        uncertainty = (
            5.0
            + 3.0 * math.sqrt(step)
        )

        confidence = max(
            0.50,
            0.95 - step * 0.08,
        )

        results.append(
            {
                "step": step,

                "predicted_delay": delay,

                "lower_delay": (
                    delay - uncertainty
                ),

                "upper_delay": (
                    delay + uncertainty
                ),

                "confidence": confidence,

                # Keep this internally useful for debugging.
                "predicted_change": change,
            }
        )

    return results