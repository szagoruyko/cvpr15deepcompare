PyTorch DeepCompare
============

So far only evaluation code is supported. Run `eval.py` script to evaluate a model on a particular subset.
You will need to use lua to convert dataset to the compatible format. `eval.py` has functional definitions
for all models, so that you can easily grab the parameters and model function and reuse it in your own code.

The numbers on Brown dataset are slightly different due to different evaluation code:

<table><tr><th>sets</th><th>siam2stream_l2</th><th>siam2stream</th><th>2ch</th><th>siam</th><th>siam_l2</th><th>2ch2stream</th></tr><tr><td>yosemite, notredame</td><td>5.63</td><td>5.37</td><td>3.05</td><td>5.76</td><td>8.40</td><td>2.22</td><tr><td>yosemite, liberty</td><td>12.02</td><td>11.03</td><td>9.02</td><td>13.58</td><td>18.90</td><td>7.48</td><tr><td>notredame, yosemite</td><td>12.85</td><td>10.29</td><td>5.73</td><td>12.63</td><td>15.18</td><td>3.99</td><tr><td>notredame, liberty</td><td>7.93</td><td>6.19</td><td>5.86</td><td>8.62</td><td>12.55</td><td>5.46</td><tr><td>liberty, yosemite</td><td>13.32</td><td>9.42</td><td>7.65</td><td>15.29</td><td>20.11</td><td>5.27</td><tr><td>liberty, notredame</td><td>5.19</td><td>3.16</td><td>3.02</td><td>4.43</td><td>6.05</td><td>1.88</td></table>

## Installation

Install PyTorch following instructions from <http://pytorch.org>,
then run:

```bash
pip install -r requirements.txt
```

And install `torchnet`:

```bash
pip install git+https://github.com/pytorch/tnt.git@master
```