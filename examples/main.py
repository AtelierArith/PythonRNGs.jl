import random
import numpy as np

def example1():
	return 2 * random.random() + 1

def example2():
	return 2 * np.random.random() + 1

def example3(rng):
	return 2 * rng.random() + 1

def main():
	random.seed(1234)
	print(f"{example1()=}")
	np.random.seed(999)
	print(f"{example2()=}")
	rng = np.random.default_rng(42)
	print(f"{example3(rng)=}")

if __name__ == "__main__":
	main()
