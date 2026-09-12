import random
import numpy as np

def example1(seed):
	random.seed(seed)
	return 2 * random.random() + 1

def example2(seed):
	np.random.seed(seed)
	return 2 * np.random.random() + 1

def example3(rng):
	return 2 * rng.random() + 1

def main():
	print(f"{example1(1234)=}")
	print(f"{example2(999)=}")
	rng = np.random.default_rng(42)
	print(f"{example3(rng)=}")

if __name__ == "__main__":
	main()
