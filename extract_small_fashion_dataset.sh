#!/bin/bash

echo "Removing existing small_fashion_mnist_images directories..."

rm -rf small_fashion_mnist_images

echo "Creating small_fashion_mnist_images directories..."

mkdir -p small_fashion_mnist_images/
mkdir -p small_fashion_mnist_images/train
mkdir -p small_fashion_mnist_images/train/0
mkdir -p small_fashion_mnist_images/train/1
mkdir -p small_fashion_mnist_images/train/2
mkdir -p small_fashion_mnist_images/train/3
mkdir -p small_fashion_mnist_images/train/4
mkdir -p small_fashion_mnist_images/train/5
mkdir -p small_fashion_mnist_images/train/6
mkdir -p small_fashion_mnist_images/train/7
mkdir -p small_fashion_mnist_images/train/8
mkdir -p small_fashion_mnist_images/train/9

echo "Copying fashion_mnist images..."

cp $(ls -1d fashion_mnist_images/train/0/* | head --lines=500) small_fashion_mnist_images/train/0/;
cp $(ls -1d fashion_mnist_images/train/1/* | head --lines=500) small_fashion_mnist_images/train/1/;
cp $(ls -1d fashion_mnist_images/train/2/* | head --lines=500) small_fashion_mnist_images/train/2/;
cp $(ls -1d fashion_mnist_images/train/3/* | head --lines=500) small_fashion_mnist_images/train/3/;
cp $(ls -1d fashion_mnist_images/train/4/* | head --lines=500) small_fashion_mnist_images/train/4/;
cp $(ls -1d fashion_mnist_images/train/5/* | head --lines=500) small_fashion_mnist_images/train/5/;
cp $(ls -1d fashion_mnist_images/train/6/* | head --lines=500) small_fashion_mnist_images/train/6/;
cp $(ls -1d fashion_mnist_images/train/7/* | head --lines=500) small_fashion_mnist_images/train/7/;
cp $(ls -1d fashion_mnist_images/train/8/* | head --lines=500) small_fashion_mnist_images/train/8/;
cp $(ls -1d fashion_mnist_images/train/9/* | head --lines=500) small_fashion_mnist_images/train/9/;

echo "Copying test dataset..."

cp -r fashion_mnist_images/test small_fashion_mnist_images/test/

echo "Done!"
# rm $(ls -d1 small_fashion_mnist_images/train/0/* | head --lines=300);
# rm $(ls -d1 small_fashion_mnist_images/train/1/* | head --lines=300);
# rm $(ls -d1 small_fashion_mnist_images/train/2/* | head --lines=300);
# rm $(ls -d1 small_fashion_mnist_images/train/3/* | head --lines=300);
# rm $(ls -d1 small_fashion_mnist_images/train/4/* | head --lines=300);
# rm $(ls -d1 small_fashion_mnist_images/train/5/* | head --lines=300);
# rm $(ls -d1 small_fashion_mnist_images/train/6/* | head --lines=300);
# rm $(ls -d1 small_fashion_mnist_images/train/7/* | head --lines=300);
# rm $(ls -d1 small_fashion_mnist_images/train/8/* | head --lines=300);
# rm $(ls -d1 small_fashion_mnist_images/train/9/* | head --lines=300);