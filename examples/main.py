import random
import numpy as np

np.set_printoptions(precision=17)

def example1(seed):
	rng = random.Random(seed)
	return [[rng.random() for _ in range(3)] for _ in range(2)]

def example2(seed):
	rng = np.random.RandomState(seed)
	return rng.random((2, 3))

def example3(rng):
	return rng.random((2, 3))

def main():
	print("example1(1234) =")
	print(example1(1234))

	print("example2(999) =")
	print(example2(999))

	rng = np.random.default_rng(42)
	print("example3(rng) =")
	print(example3(rng))

if __name__ == "__main__":
	main()
