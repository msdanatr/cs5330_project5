# CS 5330 Project 5
# Dana Tran

## description

deep learning project using pytorch + torchvision. i trained a small cnn on mnist digits, saved the trained model, reloaded it for evaluation and plotting, inspected conv1 filters + filter outputs, did transfer learning on greek letters (alpha/beta/gamma), and reimplemented the mnist classifier using a transformer model. i also added a few extensions (pretrained conv filters and live webcam demo)


## things to install

- python 3
- pytorch + torchvision
- matplotlib
- opencv-python (for task 2 + webcam extension)
- i used a venv and i activated it before running scripts.


## files
### task 1 - mnist cnn
- `task1_plot_mnist_test6.py` - saves the first 6 mnist test digits in a 2x3 grid (`plot-first-six-test-digits.png`)
- `task1_network.py` - defines `MyNetwork`, trains mnist cnn for 5 epochs, saves weights + training plots
- `task1_run_test10.py` - loads `mnist_cnn.pt`, runs first 10 mnist test samples, prints outputs/argmax/label, saves a 3x3 prediction grid (`plot-predictions.png`)
- `task1_handwritten.py` - runs the saved mnist model on my handwritten digit crops (supports `--no-invert`)
### task 2 - examine cnn
- `task2_examine.py` - prints the model + conv1 weights, saves conv1 filter visualization (`conv1_filters.png`) and filter2d results (`conv1_filterResults.png`)
### task 3 - greek transfer learning
- `task3_greek.py` - loads mnist weights, freezes the network, replaces `fc2` with 3 outputs, trains on greek letters, saves `greek_cnn.pt` + `greek-train-error.png`
- greek dataset folder: `greek_train/alpha`, `greek_train/beta`, `greek_train/gamma`
### transformer (mnist)
- `NetTransformer_template.py` - transformer model + config (forward implemented)
- `task4_train_transformer.py` - trains the transformer on mnist and saves plots + weights
  - outputs: `transformer-train-error.png`, `transformer-train-accuracy.png`, `mnist_transformer.pt`
### extensions
- `pretrained_conv.py` - loads pretrained resnet18 from torchvision and visualizes early conv filters
  - output: `pretrained_resnet18_conv1_filters.png`
- `live_digits.py` - live webcam digit recognition using `mnist_cnn.pt`
- `task5_experiment.py` - automated experiment sweep on FashionMNIST (logs results to csv)
  - output: `experiment_results.csv`
## outputs / data
- `mnist_data/` - mnist download cache (torchvision)
- `fashion_data/` - fashion-mnist download cache (torchvision)
- `handwritten/` - my handwritten digit crops (0-9)
- generated plots + `.pt` weights are saved in the project root
## use
(run these from the project 5 folder)

### task 1
python task1_plot_mnist_test6.py
python task1_network.py
python task1_run_test10.py
python task1_handwritten.py handwritten
python task1_handwritten.py handwritten --no-invert

### task 2
python task2_examine.py

### task 3
python task3_greek.py --data greek_train

###task 4
python task4_train_transformer.py

### task 5
python task5_experiment.py

### extensions
python pretrained_conv.py
python live_digits.py

## notes
handwritten digits are harder than mnist because lighting/cropping/contrast can be different so inverting and/or recropping helps

no travel days needed for this one