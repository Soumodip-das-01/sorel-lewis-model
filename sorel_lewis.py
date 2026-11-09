"""Interactive Lewis–Sorel/McCabe–Thiele calculator for a binary distillation.
Assumptions:
    * binary mixture and constant relative volatility;
    * constant molal overflow;
    * total condenser and partial/reboiler counted as equilibrium stages;
    * ideal equilibrium: y = alpha*x / (1 + (alpha - 1)*x).

The program uses the Lewis–Sorel stage-stepping idea: calculate the vapor
composition in equilibrium with the liquid leaving each stage, then use the
operating-line balance to move to the next stage.
"""

from __future__ import annotations

import pandas as pd

from dataclasses import dataclass
from math import isclose

@dataclass
class Inputs:
    feed_rate: float
    feed_fraction: float
    distillate_fraction: float
    bottoms_fraction: float
    reflux_ratio: float
    relative_volatility: float
    feed_quality: float


def read_float(label: str, *, minimum: float | None = None,
               maximum: float | None = None) -> float:
    while True:
        try:
            value = float(input(label).strip())
        except ValueError:
            print("Please enter a numeric value.")
            continue

        if minimum is not None and value < minimum:
            print(f"Value must be at least {minimum}.")
            continue
        if maximum is not None and value > maximum:
            print(f"Value must be at most {maximum}.")
            continue
        return value

def collect_inputs() -> Inputs:
    print("\nLewis–Sorel binary distillation calculator")
    print("Enter mole fractions between 0 and 1.\n")

    data = Inputs(
        feed_rate=read_float("Feed flow rate: ", minimum=0.0),
        feed_fraction=read_float("Feed mole fraction of the more volatile component: ", minimum=0.0, maximum=1.0),
        distillate_fraction=read_float("Distillate mole fraction: ", minimum=0.0, maximum=1.0),
        bottoms_fraction=read_float("Bottoms mole fraction: ", minimum=0.0, maximum=1.0),
        reflux_ratio=read_float("Reflux ratio (R): ", minimum=0.0),
        relative_volatility=read_float("Relative volatility (alpha, > 1): ", minimum=1.000001),
        feed_quality=read_float("Feed quality, q (usually 0 to 1): ", minimum=0.0),
    )

    if not data.bottoms_fraction < data.feed_fraction < data.distillate_fraction:
        raise ValueError("Expected bottoms fraction < feed fraction < distillate fraction.")
    if isclose(data.distillate_fraction, data.bottoms_fraction):
        raise ValueError("Distillate and bottoms compositions must be different.")
    return data


def equilibrium_y(x: float, alpha: float) -> float:
    return alpha * x / (1.0 + (alpha - 1.0) * x)


def calculate(data: input) -> dict[str, float | int ]:
    f, zf, xd, xb, reflux, alpha, q = (
        data.feed_rate,
        data.feed_fraction,
        data.distillate_fraction,
        data.bottoms_fraction,
        data.reflux_ratio,
        data.relative_volatility,
        data.feed_quality,
    )

     # Overall and component balances.
    distillate = f * (zf - xb) / (xd - xb)
    bottoms = f - distillate

     # Rectifying operating line: y = m_R*x + c_R.
    m_rect = reflux / (reflux + 1.0)
    c_rect = xd / (reflux + 1.0)

    if isclose(q, 1.0):
        x_intersection = zf
        y_intersection = m_rect * x_intersection + c_rect

    else:
        m_q = q / (q - 1.0)
        c_q = -zf / (q - 1.0)
        x_intersection = (c_q - c_rect) / (m_rect - m_q)
        y_intersection = m_rect * x_intersection + c_rect

    # Stripping line through (xb, xb) and the feed-line intersection.
    m_strip = (y_intersection - xb) / (x_intersection - xb)
    c_strip = xb * (1.0 - m_strip)

    # Step off stages from the top composition toward the reboiler.
    y_current = xd
    stages = 0
    stage_rows: list[tuple[int, float, float, str]] = []
    max_stages = 1000

    while y_current > xb and stages < max_stages:
        stages += 1
        # Horizontal move: equilibrium relation y -> x.
        x_liquid = y_current / (alpha - (alpha - 1.0) * y_current)
        if x_liquid <= xb:
            stage_rows.append((stages, y_current, x_liquid, "reboiler"))
            break

        # Vertical move: choose the operating line based on the feed location.
        if x_liquid >= x_intersection:
            y_next = m_rect * x_liquid + c_rect
            section = "rectifying"
        else:
            y_next = m_strip * x_liquid + c_strip
            section = "stripping"

        stage_rows.append((stages, y_current, x_liquid, section))
        if y_next >= y_current:
            raise ValueError("Stage stepping failed; check compositions and operating conditions.")
        y_current = y_next

    if stages >= max_stages:
        raise ValueError("The calculation did not converge within 1000 stages.")

    return {
        "distillate_rate": distillate,
        "bottoms_rate": bottoms,
        "feed_intersection_x": x_intersection,
        "feed_intersection_y": y_intersection,
        "rectifying_slope": m_rect,
        "stripping_slope": m_strip,
        "theoretical_stages": stages,
        "stage_rows": stage_rows,
    }


def print_results(data: Inputs, results: dict[str, float | int]) -> None:
    print("\nResults")
    print("-" * 42)
    print(f"Distillate flow rate : {results['distillate_rate']:.4f}")
    print(f"Bottoms flow rate    : {results['bottoms_rate']:.4f}")
    print(f"Feed intersection x  : {results['feed_intersection_x']:.4f}")
    print(f"Feed intersection y  : {results['feed_intersection_y']:.4f}")
    print(f"Rectifying slope     : {results['rectifying_slope']:.4f}")
    print(f"Stripping slope      : {results['stripping_slope']:.4f}")
    print(f"Theoretical stages   : {results['theoretical_stages']}")
    stage_table = pd.DataFrame(
        results["stage_rows"],
        columns=["Stage", "Vapor_y_in", "Liquid_x", "Section"],
    )
    stage_table["Vapor_y_in"] = stage_table["Vapor_y_in"].round(5)
    stage_table["Liquid_x"] = stage_table["Liquid_x"].round(5)
    print("\nStage details")
    print(stage_table.to_string(index=False))

    output_file = "stage_results.csv"
    stage_table.to_csv(output_file, index=False)
    print(f"\nStage results saved to {output_file}")


def main() -> None:
    try:
        data = collect_inputs()
        print_results(data, calculate(data))
    except ValueError as error:
        print(f"\nInput/calculation error: {error}")


if __name__ == "__main__":
    main()