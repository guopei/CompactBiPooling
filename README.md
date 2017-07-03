# Compact Bilinear Pooling for Torch 

This code is revised from [@jnhwkim's](https://github.com/jnhwkim) [Multimodal Compact Bilinear Pooling for Torch7](https://github.com/jnhwkim/cbp). The main changes include:

1. adapt the code with image inputs(4D tensor).
2. get rid of spectral-lib dependency.
3. the new package name is **tcbp**(Torch Compact Bilinear Pooling) to avoid confusion.
4. new tests.

The compact bilinear pooling layer is proposed by Yang Gao etc. in the paper [Compact Bilinear Pooling](https://arxiv.org/abs/1511.06062). This method reduces the spatial complexity of [Bilinear Pooling](http://vis-www.cs.umass.edu/bcnn/docs/bcnn_iccv15.pdf) so that it's feasible for real world training and provides a possible direction to interpret the huge success in fine grained recognition using Bilibear Pooling. We refer you to [caffe implementation page](https://github.com/gy20073/compact_bilinear_pooling) for further information.

## Installation

```
git clone https://github.com/guopei/CompactBiPooling cbp
cd cbp
luarocks make rocks/tcbp-1.0-1.rockspec
```

## Test

```
th test.lua
```
Read test.lua for usage.

## Troubleshooting

If the following error occurs:
```
ld -lcufft not found
```
This is because the `ld` cannot find the `cufft` library even if you have already set the `LD_LIBRARY_PATH`. 
Because `LD_LIBRARY_PATH` is only used at excution time not compilation time.
The solution is just adding a soft link of `libcufft.so` to your system library path:
```
sudo ln -s /user/local/cuda/lib64/libcufft.so /user/lib
``` 
Change the location of `libcufft.so` according to where its' located in your machine.

## Update

1. Added code to be compatible with both lua51 and lua52.

## References
1. [spectral-lib](https://github.com/mbhenaff/spectral-lib)
2. [cbp](https://github.com/jnhwkim/cbp)
3. [compact_bilinear_pooling](https://github.com/gy20073/compact_bilinear_pooling)
4. [tensorflow_compact_bilinear_pooling](https://github.com/ronghanghu/tensorflow_compact_bilinear_pooling)
