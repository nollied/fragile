# import numpy
import pytest

from fragile.backend import dtype, tensor
from fragile.core.bounds import Bounds


def create_bounds(name):
    if name == "scalars":
        return lambda: Bounds(high=5, low=-5, shape=(3,))
    elif name == "high_array":
        return lambda: Bounds(high=tensor([1, 2, 5], dtype=dtype.float), low=-5)
    elif name == "low_array":
        return lambda: Bounds(low=tensor([-1, -5, -3], dtype=dtype.float), high=5)
    elif name == "both_array":
        array = tensor([1, 2, 5], dtype=dtype.float)
        return lambda: Bounds(high=array, low=-array)
    elif name == "high_list":
        return lambda: Bounds(
            low=tensor([-5, -2, -3], dtype=dtype.float), high=[5, 5, 5], dtype=dtype.float
        )


bounds_fixture_params = ["scalars", "high_array", "low_array", "both_array", "high_list"]


@pytest.fixture(params=bounds_fixture_params, scope="class")
def bounds_fixture(request) -> Bounds:
    return create_bounds(request.param)()


class TestBounds:
    def test_init(self, bounds_fixture):
        assert bounds_fixture.dtype is not None
        assert dtype.is_tensor(bounds_fixture.low)
        assert dtype.is_tensor(bounds_fixture.high)
        assert isinstance(bounds_fixture.shape, tuple)
        assert bounds_fixture.low.shape == bounds_fixture.shape
        assert bounds_fixture.high.shape == bounds_fixture.shape

    def test_shape(self, bounds_fixture: Bounds):
        shape = bounds_fixture.shape
        assert isinstance(shape, tuple)
        assert shape == (3,)

    def test_points_in_bounds(self, bounds_fixture):
        points = tensor([[0, 0, 0], [11, 0, 0], [0, 11, 0], [11, 11, 11]])
        res = bounds_fixture.points_in_bounds(points)
        for a, b in zip(res.tolist(), [True, False, False, False]):
            assert a == b

    def test_from_tuples(self):
        tup = ((-1, 2), (-3, 4), (2, 5))
        bounds = Bounds.from_tuples(tup)
        assert (bounds.low == tensor([-1, -3, 2])).all()
        assert (bounds.high == tensor([2, 4, 5])).all()

    def test_from_array(self):
        array = tensor([[0, 0, 0], [11, 0, 0], [0, 11, 0], [11, 11, 11]])
        bounds = Bounds.from_array(array)
        assert (bounds.low == tensor([0, 0, 0])).all()
        assert (bounds.high == tensor([11, 11, 11])).all()
        assert bounds.shape == (3,)

    def test_from_array_with_scale_positive(self):
        array = tensor([[0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 10]], dtype=dtype.float)
        bounds = Bounds.from_array(array, scale=1.1)
        assert (bounds.low == tensor([0, 0, 0], dtype=dtype.float)).all(), (
            bounds.low,
            array.min(axis=0),
        )
        assert (bounds.high == tensor([11, 11, 11], dtype=dtype.float)).all(), (
            bounds.high,
            array.max(axis=0),
        )
        assert bounds.shape == (3,)

        array = tensor([[-10, 0, 0], [-10, 0, 0], [0, -10, 0], [-10, -10, -10]], dtype=dtype.float)
        bounds = Bounds.from_array(array, scale=1.1)
        assert (bounds.high == tensor([0, 0, 0], dtype=dtype.float)).all(), (
            bounds.high,
            array.max(axis=0),
        )
        assert (bounds.low == tensor([-11, -11, -11], dtype=dtype.float)).all(), (
            bounds.low,
            array.min(axis=0),
        )
        assert bounds.shape == (3,)

        array = tensor(
            [[10, 10, 10], [100, 10, 10], [10, 100, 10], [100, 100, 100]], dtype=dtype.float
        )
        bounds = Bounds.from_array(array, scale=1.1)
        assert tensor.allclose(bounds.low, tensor([9.0, 9.0, 9], dtype=dtype.float)), (
            bounds.low,
            array.min(axis=0),
        )
        assert tensor.allclose(bounds.high, tensor([110, 110, 110], dtype=dtype.float)), (
            bounds.high,
            array.max(axis=0),
        )
        assert bounds.shape == (3,)

    def test_from_array_with_scale_negative(self):
        # high +, low +, scale > 1
        array = tensor([[-10, 0, 0], [-10, 0, 0], [0, -10, 0], [-10, -10, -10]], dtype=dtype.float)
        bounds = Bounds.from_array(array, scale=0.9)
        assert (bounds.high == tensor([0, 0, 0], dtype=dtype.float)).all(), (
            bounds.high,
            array.max(axis=0),
        )
        assert (bounds.low == tensor([-9, -9, -9], dtype=dtype.float)).all(), (
            bounds.low,
            array.min(axis=0),
        )
        assert bounds.shape == (3,)
        array = tensor([[0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 10]], dtype=dtype.float)
        bounds = Bounds.from_array(array, scale=0.9)
        assert (bounds.low == tensor([0, 0, 0], dtype=dtype.float)).all(), (bounds, array)
        assert (bounds.high == tensor([9, 9, 9], dtype=dtype.float)).all()
        assert bounds.shape == (3,)
        # high +, low +, scale < 1
        array = tensor(
            [[10, 10, 10], [100, 10, 10], [10, 100, 10], [100, 100, 100]], dtype=dtype.float
        )
        bounds = Bounds.from_array(array, scale=0.9)
        assert tensor.allclose(bounds.low, tensor([9, 9, 9], dtype=dtype.float)), (
            bounds.low,
            array.min(axis=0),
        )
        assert tensor.allclose(bounds.high, tensor([90, 90, 90], dtype=dtype.float)), (
            bounds.high,
            array.max(axis=0),
        )
        assert bounds.shape == (3,)
        # high -, low -, scale > 1
        array = tensor(
            [[-100, -10, -10], [-100, -10, -10], [-10, -100, -10], [-100, -100, -100]],
            dtype=dtype.float,
        )
        bounds = Bounds.from_array(array, scale=1.1)
        assert tensor.allclose(bounds.high, tensor([-9, -9, -9], dtype=dtype.float)), (
            bounds.high,
            array.max(axis=0),
        )
        assert tensor.allclose(bounds.low, tensor([-110, -110, -110], dtype=dtype.float)), (
            bounds.low,
            array.min(axis=0),
        )
        assert bounds.shape == (3,)
        # high -, low -, scale < 1
        array = tensor(
            [[-100, -10, -10], [-100, -10, -10], [-10, -100, -10], [-100, -100, -100]],
            dtype=dtype.float,
        )
        bounds = Bounds.from_array(array, scale=0.9)
        assert tensor.allclose(bounds.high, tensor([-11, -11, -11], dtype=dtype.float)), (
            bounds.high,
            array.max(axis=0),
        )
        assert tensor.allclose(bounds.low, tensor([-90, -90, -90], dtype=dtype.float)), (
            bounds.low,
            array.min(axis=0),
        )
        assert bounds.shape == (3,)

    def test_clip(self):
        tup = ((-1, 10), (-3, 4), (2, 5))
        array = tensor([[-10, 0, 0], [11, 0, 0], [0, 11, 0], [11, 11, 11]], dtype=dtype.float)
        bounds = Bounds.from_tuples(tup)
        clipped = bounds.clip(array)
        target = tensor(
            [[-1.0, 0.0, 2.0], [10.0, 0.0, 2.0], [0.0, 4.0, 2], [10, 4, 5]], dtype=dtype.float
        )
        print("dtypes", clipped.dtype, target.dtype)
        assert tensor.allclose(clipped, target), (clipped.dtype, target.dtype)

    @pytest.mark.parametrize("bounds_fixture", bounds_fixture_params, indirect=True)
    def test_to_tuples(self, bounds_fixture):
        tuples = bounds_fixture.to_tuples()
        assert len(tuples) == 3
        assert min([x[0] for x in tuples]) == -5
        assert max([x[1] for x in tuples]) == 5

    @pytest.mark.parametrize("bounds_fixture", bounds_fixture_params, indirect=True)
    def test_points_in_bounds(self, bounds_fixture):
        zeros = tensor.zeros((3, 3))
        assert all(bounds_fixture.points_in_bounds(zeros))
        tens = tensor.ones((3, 3)) * 10.0
        res = bounds_fixture.points_in_bounds(tens)
        assert not res.any(), (res, tens)
        tens = tensor([[-10, 0, 1], [0, 0, 0], [10, 10, 10]])
        assert sum(bounds_fixture.points_in_bounds(tens)) == 1

    @pytest.mark.parametrize("bounds_fixture", bounds_fixture_params, indirect=True)
    def test_safe_margin(self, bounds_fixture: Bounds):
        new_bounds = bounds_fixture.safe_margin()
        assert tensor.allclose(new_bounds.low, bounds_fixture.low)
        assert tensor.allclose(new_bounds.high, bounds_fixture.high)
        low = tensor.full_like(bounds_fixture.low, -10)
        new_bounds = bounds_fixture.safe_margin(low=low)
        assert tensor.allclose(new_bounds.high, bounds_fixture.high)
        assert tensor.allclose(new_bounds.low, low)
        new_bounds = bounds_fixture.safe_margin(low=low, scale=2)
        assert tensor.allclose(new_bounds.high, bounds_fixture.high * 2)
        assert tensor.allclose(new_bounds.low, low * 2)
