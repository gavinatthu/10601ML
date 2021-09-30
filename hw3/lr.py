import sys


def main():
    return None

if __name__ == '__main__':
    '''
    python lr.py formatted_train.tsv formatted_valid.tsv formatted_test.tsv dict.txt train_out.labels test_out.labels metrics_out.txt 60
    '''

    train_in = sys.argv[1]
    test_in = sys.argv[2]
    max_depth = int(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics = sys.argv[6]
    main(train_in, test_in, max_depth, train_out, test_out, metrics)