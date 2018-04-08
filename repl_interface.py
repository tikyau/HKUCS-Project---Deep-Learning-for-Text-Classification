import sys

from predictor import Predictor


def main():
    predictor = Predictor(sys.argv[1], sys.argv[2], sys.argv[3])
    while True:
        try:
            inp = input(">> ")
        except UnicodeError:
            continue
        if inp == ":q":
            break
        if inp:
            res = predictor.predict(inp)
            print("Words: {}\nScore: {}".format(res[0], res[1][0] + 1))


if __name__ == "__main__":
    main()
