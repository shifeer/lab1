import numpy as np
import pytest

from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixVectorBatchInput,
    OneHotInput, RandomMatrixInput, RectangleInput, TimeSeriesInput,
)
from numpy_tasks import (
    analyze_time_series, binarize, chess, draw_ellipse, draw_rectangle,
    matrix_statistics, one_hot, sum_prod, unique_columns, unique_rows,
)

RED, BLACK = (255, 0, 0), (0, 0, 0)


# Задача 1
def test_sum_prod_matches_loop():
    rng = np.random.default_rng(0)
    matrices = rng.integers(-5, 5, size=(4, 3, 3))
    vectors = rng.integers(-5, 5, size=(4, 3, 1))
    expected = sum(a @ x for a, x in zip(matrices, vectors))
    result = sum_prod(MatrixVectorBatchInput(matrices, vectors))
    assert result.shape == (3, 1)
    np.testing.assert_array_equal(result, expected)


def test_sum_prod_lists_and_identity():
    matrices = [np.eye(2), 2 * np.eye(2)]
    vectors = [np.array([[1], [2]]), np.array([[3], [4]])]
    np.testing.assert_array_equal(
        sum_prod(MatrixVectorBatchInput(matrices, vectors)), [[7], [10]])


def test_sum_prod_single_and_1x1():
    result = sum_prod(MatrixVectorBatchInput([[[3]]], [[[4]]]))
    np.testing.assert_array_equal(result, [[12]])


def test_sum_prod_mismatch():
    with pytest.raises(ValueError):
        sum_prod(MatrixVectorBatchInput([np.eye(2)], []))


# Задача 2
def test_binarize_strict_and_copy():
    matrix = np.array([[0.1, 0.5, 0.9], [0.5, 0.51, -1.0]])
    original = matrix.copy()
    result = binarize(BinarizeInput(matrix))
    np.testing.assert_array_equal(result, [[0, 0, 1], [0, 1, 0]])
    np.testing.assert_array_equal(matrix, original)
    assert result is not matrix


def test_binarize_custom_threshold_and_edges():
    np.testing.assert_array_equal(binarize(BinarizeInput([[1, 2], [3, 4]], 2)), [[0, 0], [1, 1]])
    np.testing.assert_array_equal(binarize(BinarizeInput([[5, 5]], 5)), [[0, 0]])
    assert binarize(BinarizeInput(np.empty((0, 3)))).shape == (0, 3)


# Задача 3
def test_unique_rows_and_columns():
    matrix = np.array([[1, 2, 2], [3, 3, 3], [1, 2, 4]])
    assert unique_rows(MatrixInput(matrix)) == [[1, 2], [3], [1, 2, 4]]
    assert unique_columns(MatrixInput(matrix)) == [[1, 3], [2, 3], [2, 3, 4]]


def test_unique_single_row_and_column():
    assert unique_rows(MatrixInput([[5, 5, 1]])) == [[1, 5]]
    assert unique_columns(MatrixInput([[5, 5, 1]])) == [[5], [5], [1]]
    assert unique_rows(MatrixInput([[1.5], [1.5]])) == [[1.5], [1.5]]


# Задача 4
def test_matrix_statistics_shapes_and_values():
    stats = matrix_statistics(RandomMatrixInput(rows=3, columns=4, mean=5, std=2, seed=1))
    m = stats.matrix
    assert m.shape == (3, 4)
    np.testing.assert_allclose(stats.row_means, m.mean(axis=1))
    np.testing.assert_allclose(stats.column_means, m.mean(axis=0))
    np.testing.assert_allclose(stats.row_variances, m.var(axis=1))
    np.testing.assert_allclose(stats.column_variances, m.var(axis=0))


def test_matrix_statistics_seed_and_distribution():
    a = matrix_statistics(RandomMatrixInput(200, 300, mean=10, std=3, seed=7))
    b = matrix_statistics(RandomMatrixInput(200, 300, mean=10, std=3, seed=7))
    c = matrix_statistics(RandomMatrixInput(200, 300, mean=10, std=3, seed=8))
    np.testing.assert_array_equal(a.matrix, b.matrix)
    assert not np.array_equal(a.matrix, c.matrix)
    assert a.matrix.mean() == pytest.approx(10, abs=0.05)
    assert a.matrix.std() == pytest.approx(3, abs=0.05)


def test_matrix_statistics_zero_std():
    stats = matrix_statistics(RandomMatrixInput(2, 2, mean=1, std=0))
    np.testing.assert_array_equal(stats.matrix, np.ones((2, 2)))
    np.testing.assert_array_equal(stats.row_variances, [0, 0])


# Задача 5
def test_chess_example():
    expected = [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]
    np.testing.assert_array_equal(chess(ChessInput(3, 4, 0, 1)), expected)


def test_chess_edges():
    np.testing.assert_array_equal(chess(ChessInput(1, 1, 7, 9)), [[7]])
    np.testing.assert_array_equal(chess(ChessInput(1, 3, 7, 9)), [[7, 9, 7]])
    np.testing.assert_array_equal(chess(ChessInput(3, 1, 7, 9)), [[7], [9], [7]])
    np.testing.assert_array_equal(chess(ChessInput(2, 2, 0.5, -1)), [[0.5, -1], [-1, 0.5]])


# Задача 6
def test_draw_rectangle_centered():
    image = draw_rectangle(RectangleInput(4, 2, 6, 8, RED, BLACK))
    assert image.shape == (6, 8, 3)
    mask = np.all(image == RED, axis=2)
    rows, cols = np.nonzero(mask)
    assert mask.sum() == 8
    assert (rows.min(), rows.max(), cols.min(), cols.max()) == (2, 3, 2, 5)
    assert np.all(image[~mask] == BLACK)


def test_draw_rectangle_edges():
    full = draw_rectangle(RectangleInput(5, 5, 5, 5, RED, BLACK))
    assert np.all(full == RED)
    empty = draw_rectangle(RectangleInput(0, 0, 4, 4, RED, BLACK))
    assert np.all(empty == BLACK)


def test_draw_ellipse():
    image = draw_ellipse(EllipseInput(3, 2, 11, 11, RED, BLACK))
    assert image.shape == (11, 11, 3)
    mask = np.all(image == RED, axis=2)
    assert mask[5, 5]                       # центр
    assert mask[5, 2] and mask[5, 8]        # концы оси x
    assert mask[3, 5] and mask[7, 5]        # концы оси y
    assert not mask[5, 1] and not mask[2, 5]
    assert not mask[0, 0]
    np.testing.assert_array_equal(mask, mask[::-1, ::-1])  # симметрия


def test_draw_ellipse_circle_and_degenerate():
    mask = np.all(draw_ellipse(EllipseInput(1, 1, 3, 3, RED, BLACK)) == RED, axis=2)
    np.testing.assert_array_equal(mask, [[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    assert np.all(draw_ellipse(EllipseInput(0, 3, 5, 5, RED, BLACK)) == BLACK)


# Задача 7
def test_analyze_time_series():
    values = [1, 3, 2, 2, 5, 1, 4]
    stats = analyze_time_series(TimeSeriesInput(values, 3))
    arr = np.array(values, dtype=float)
    assert stats.mean == pytest.approx(arr.mean())
    assert stats.variance == pytest.approx(arr.var())
    assert stats.std == pytest.approx(arr.std())
    np.testing.assert_array_equal(stats.local_maxima_indices, [1, 4])
    np.testing.assert_array_equal(stats.local_minima_indices, [5])
    np.testing.assert_allclose(stats.moving_average, [2, 7 / 3, 3, 8 / 3, 10 / 3])
    assert len(stats.moving_average) == len(values) - 3 + 1


def test_analyze_time_series_edges():
    flat = analyze_time_series(TimeSeriesInput([2, 2, 2], 1))
    assert flat.variance == 0 and flat.std == 0
    assert len(flat.local_maxima_indices) == 0 and len(flat.local_minima_indices) == 0
    np.testing.assert_array_equal(flat.moving_average, [2, 2, 2])

    single = analyze_time_series(TimeSeriesInput([5], 1))
    assert single.mean == 5 and len(single.local_maxima_indices) == 0

    whole = analyze_time_series(TimeSeriesInput([1, 2, 3, 4], 4))
    np.testing.assert_allclose(whole.moving_average, [2.5])

    with pytest.raises(ValueError):
        analyze_time_series(TimeSeriesInput([1, 2], 3))
    with pytest.raises(ValueError):
        analyze_time_series(TimeSeriesInput([1, 2], 0))


# Задача 8
def test_one_hot_example():
    expected = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]
    np.testing.assert_array_equal(one_hot(OneHotInput([0, 2, 3, 0])), expected)


def test_one_hot_class_count_and_edges():
    result = one_hot(OneHotInput([1, 0], class_count=5))
    np.testing.assert_array_equal(result, [[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]])
    np.testing.assert_array_equal(one_hot(OneHotInput([0, 0])), [[1], [1]])
    assert one_hot(OneHotInput([], class_count=3)).shape == (0, 3)
    with pytest.raises(ValueError):
        one_hot(OneHotInput([0, 3], class_count=2))
    with pytest.raises(ValueError):
        one_hot(OneHotInput([-1, 0]))
