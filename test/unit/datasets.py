
from bspytasks.datasets.boolean import VCDimensionDataset
from bspytasks.datasets.ring import RingDataGenerator


if __name__ == '__main__':
    from bspytasks.utils.transforms as utils
    transforms = transforms.Compose([utils.ToTensor()])
    dataset = RingDataGenerator(1000, 0.2)

    # print(dataset.__getitem__(0)['targets'].shape)

    for i in range(dataset.__len__()):
        print(dataset.__getitem__(i)['inputs'])
        print(dataset.__getitem__(i)['targets'])

    print('Dataset length: ')
    print(dataset.__len__())

    print('Input type: ')
    print(type(dataset.__getitem__(0)['inputs']))

    print('Input shape: ')
    print(dataset.__getitem__(0)['inputs'].shape)

    print('Target type: ')
    print(type(dataset.__getitem__(0)['targets']))

    print('Target shape: ')
    print(dataset.__getitem__(0)['targets'].shape)
