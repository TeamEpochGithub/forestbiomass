from collections import Counter


def create_coordinates():
    """
    Returns 9 evenly spaced horizontal coordinate lines across a 256, 256 image
    """
    center_coordinates = [(int(256 / 9) * x) - 13 for x in range(1, 10, 3)]

    left = [center_coordinates[0] + x for x in range(1, 26)]
    middle = [center_coordinates[1] + x for x in range(1, 26)]
    right = [center_coordinates[2] + x for x in range(1, 26)]

    all_horizontal = left + middle + right
    combinations = [(x, y) for y in center_coordinates for x in all_horizontal]

    combinations_chunks = [
        combinations[x : x + 25] for x in range(0, len(combinations), 25)
    ]
    return combinations_chunks


def is_corrupted(band):
    """
    Returns if band is corrupted by checking if pixels in >= 3 out of 9 lines are all the same value
    """
    chunk_coordinates = create_coordinates()

    corrupted_count = 0
    for chunk in chunk_coordinates:
        coordinate_values = []

        for coordinate in chunk:
            coordinate_values.append(band[coordinate])

        c = Counter(coordinate_values)
        pixel_counts = c.most_common()
        threshold = 25

        if pixel_counts[0][1] >= threshold:
            corrupted_count += 1

    return corrupted_count >= 3
