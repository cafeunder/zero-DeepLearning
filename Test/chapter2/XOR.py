from chapter2.AND import AND
from chapter2.OR import OR
from chapter2.NAND import NAND

def XOR(x1, x2):
	return AND(NAND(x1, x2), OR(x1, x2))

if __name__ == '__main__':
	print(XOR(0, 0))
	print(XOR(0, 1))
	print(XOR(1, 0))
	print(XOR(1, 1))
