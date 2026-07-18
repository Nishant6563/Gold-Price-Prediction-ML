from src.generate_data import generate


def test_generator_is_reproducible():
    first = generate(rows=20, seed=7)
    second = generate(rows=20, seed=7)
    assert first.equals(second)
    assert first.isna().sum().sum() == 0
    assert (first["gold_usd"] > 0).all()

