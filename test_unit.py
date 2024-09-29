from check import square, cube

def test_square():
    x = 2
    res = square(x)
    assert res == 4
    assert x == 2

def helper():
    assert 1 == 1

def test_cube():
    x = 2
    res = cube(x)
    assert res == 8
