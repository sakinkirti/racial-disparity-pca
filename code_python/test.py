import h5py

def main():
    f1path = "/Users/sakinkirti/Programming/Python/CCIPD/racial-disparity-pca/dataset/model-outputs/racial-disparity-hdf5/train.h5"

    f1 = h5py.File(f1path, 'r', libver='latest')

    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()