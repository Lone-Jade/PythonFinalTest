import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def exp_norm(x, k):
    return (math.exp(k * x) - 1.0) / (math.exp(k) - 1.0)


def map_point(x, y, x_min, x_max, y_min, y_max, left, top, width, height):
    px = left + (x - x_min) / (x_max - x_min) * width
    py = top + height - (y - y_min) / (y_max - y_min) * height
    return px, py


def draw_axes(draw, box, x_label, y_label, title, font, title_font):
    left, top, width, height = box
    right = left + width
    bottom = top + height
    draw.line((left, bottom, right, bottom), fill=(35, 35, 35), width=2)
    draw.line((left, top, left, bottom), fill=(35, 35, 35), width=2)
    for i in range(6):
        x = left + i * width / 5
        y = top + i * height / 5
        draw.line((x, bottom - 4, x, bottom + 4), fill=(70, 70, 70), width=1)
        draw.line((left - 4, y, left + 4, y), fill=(70, 70, 70), width=1)
        draw.line((x, top, x, bottom), fill=(225, 225, 225), width=1)
        draw.line((left, y, right, y), fill=(225, 225, 225), width=1)
    draw.text((left + width / 2 - 80, bottom + 32), x_label, fill=(30, 30, 30), font=font)
    draw.text((left - 62, top - 28), y_label, fill=(30, 30, 30), font=font)
    draw.text((left + 16, top - 58), title, fill=(20, 20, 20), font=title_font)


def draw_polyline(draw, points, color, width=4):
    for a, b in zip(points, points[1:]):
        draw.line((a, b), fill=color, width=width)


def main():
    out = Path("fatigue_functions.png")

    img = Image.new("RGB", (1500, 620), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    title_font = ImageFont.load_default(size=22)

    # Parameters for illustration.
    f0 = 0.20
    alpha = 0.035
    f_force = 0.80
    f_resume = 0.50

    beta = 0.60
    theta = 0.60
    k = 12.0

    box1 = (95, 120, 560, 390)
    box2 = (850, 120, 560, 390)

    draw_axes(
        draw,
        box1,
        "continuous working time p",
        "fatigue F",
        "Fatigue growth during work",
        font,
        title_font,
    )
    draw_axes(
        draw,
        box2,
        "fatigue F",
        "actual_p / base_p",
        "Processing-time multiplier",
        font,
        title_font,
    )

    left, top, width, height = box1
    points = []
    for p in range(0, 121):
        f = 1.0 - (1.0 - f0) * math.exp(-alpha * p)
        points.append(map_point(p, f, 0, 120, 0, 1, left, top, width, height))
    draw_polyline(draw, points, (37, 99, 235))

    for y_value, color, label in [
        (f_force, (220, 38, 38), "F_force = 0.8"),
        (f_resume, (22, 163, 74), "F_resume = 0.5"),
    ]:
        _, y = map_point(0, y_value, 0, 120, 0, 1, left, top, width, height)
        draw.line((left, y, left + width, y), fill=color, width=2)
        draw.text((left + width - 120, y - 18), label, fill=color, font=font)

    left, top, width, height = box2
    points = []
    for i in range(301):
        f = i / 300
        sigmoid = 1.0 / (1.0 + math.exp(-k * (f - theta)))
        factor = 1.0 + beta * sigmoid
        points.append(map_point(f, factor, 0, 1, 0.95, 1.70, left, top, width, height))
    draw_polyline(draw, points, (124, 58, 237))

    for x_value, color, label in [
        (theta, (249, 115, 22), "theta = 0.6"),
        (f_force, (220, 38, 38), "F_force = 0.8"),
    ]:
        x, _ = map_point(x_value, 0.95, 0, 1, 0.95, 1.70, left, top, width, height)
        draw.line((x, top, x, top + height), fill=color, width=2)
        draw.text((x + 6, top + 12), label, fill=color, font=font)

    draw.text(
        (95, 545),
        "Fatigue: F' = 1 - (1 - F0) * exp(-alpha * p).  Example: F0=0.2, alpha=0.035.",
        fill=(50, 50, 50),
        font=font,
    )
    draw.text(
        (850, 545),
        "Time multiplier: actual_p/base_p = 1 + beta * sigmoid(k*(F-theta)).  Example: beta=0.6, k=12.",
        fill=(50, 50, 50),
        font=font,
    )

    img.save(out)
    print(out.resolve())


if __name__ == "__main__":
    main()
