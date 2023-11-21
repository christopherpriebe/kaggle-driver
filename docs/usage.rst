=====
Usage
=====

The Kaggle Driver implements boiler plate code for Kaggle competitions so that you can focus on the fun stuff.
This package can be used to easily create a command line interface for Kaggle Competition model development.
.. If you would prefer your development to be completely encapsulated in Python scripts/software or Jupyter Notebooks, the Kaggle Driver also supports this.

----------------------------
Kaggle Driver as CLI Builder
----------------------------
The Kaggle Driver can be used to create an easy-to-use command line interface for your Kaggle Competition model development.
The CLI will allow you to download the dataset, train/test various models, and create a submission file for the competition.

At a very high-level, let's go through the basic steps to create a CLI for your Kaggle Competition model development.
Before you start, remember to import the package in all Python files::

    import kaggle_driver as kd

The goal of any machine learning model is to approximate an underlying function that maps input features to a target variable.
Therefore, the first step in creating a CLI for your Kaggle Competition model development is to define class a for the input features and a class for the target variable.
These classes should inherit from the ``kaggle_driver.Input`` and ``kaggle_driver.Target`` classes, respectively.
For example, if your input features are images and you are performing classification, you might define the following classes::

    class Image(kd.Input):
        def __init__(self, image):
            self.image = image


    class Label(kd.Target):
        def __init__(self, label):
            self.label = label

Notice that these classes can define any underlying behavior that is desired by the user such as format conversion/checking (i.e., if the image was passed as a list converting to a NumPy NDArray and ensuring all values are floats in a certain range).
The behaviour for converting the `Input`` objects into valid model inputs and the model outputs into `Target` objects will be defined later.

Next, you will need to define a class to describe the dataset for the competition.
As you might expect, this class will inherit from the ``kaggle_driver.Dataset`` class::

    class MyDataset(kd.Dataset):
        ...

Unlike the ``Input`` and ``Target`` classes, the ``Dataset`` class requires the user to implement abstract methods.
These methods are ``load_train``, ``load_test``, and ``store_predictions``.
For examples on how each of these methods are implemented for various Kaggle Competitions, see :ref:`Examples<>`.

Let's look at the first method to be implemented::

    def load_train(self) -> OrderedDict[str, tuple[Input, Target]]:
        ...

This method loads the training data from some source and constructs and ordered mapping from training example names to the input features and target value for that example.

The next method to be implemented is::

    def load_test(self) -> OrderedDict[str, Input]:
        ...

This method loads the test data from some source and constructs and ordered mapping from test example names to the input features for that example.

The final method to be implemented is::

    def store_predictions(self, predictions_file: str, predictions: dict[str, Target]) -> None:
        ...

This method uses the file path to the submission file and the mapping from test example names to target values to create a submission file for the competition.

Now that you have defined the classes for the input features, target values, and dataset, you can define the class for the model.
This class will inherit from the ``kaggle_driver.Model`` class.
When working towards a solution for a Kaggle Competition, you will likely want to try many different model architectures where each model architecture can be parametrized by many hyperparameters.
Kaggle Driver supports this by allowing you to define parametrizable classes for each model architecture and registering the model classes with an internal directory.
Then, during training/testing, you will only need to pass the name of the model architecture and the hyperparameter values to use and the rest will be done for you.
Let's look at an example of how to define a parametrizable class for a model architecture::

    @kd.model
    class MyModel(kd.Model):
        def __init__(self, param1: int, param2: float):
            super().__init__("my_model")

            self.check_param1(param1)
            self.check_param2(param2)

            self.param1 = param1
            self.param2 = param2

            self.model = ...

        def check_param1(self, param1: int) -> None:
            if param1 < 0:
                raise ValueError("param1 must be non-negative")

        def check_param2(self, param2: float) -> None:
            if param2 < 0.0 or param2 > 1.0:
                raise ValueError("param2 must be in the range [0.0, 1.0]")

        ...

This model, which I have named `my_model` by passing the string `"my_model"` to the ``__init__`` method of the ``Model`` class, has two parameters, ``param1`` and ``param2``, that can be used during training and testing.
The ``check_param1`` and ``check_param2`` methods are used to check that the values passed to the parameters are valid.
While it is not stricly necessary to check the values of the parameters, it is recommended to do so to avoid errors later on.
The model class also contains a ``model`` attribute that is used to store the actual model architecture.
There are two additional methods that must be implemented by the user: ``train`` and ``test``.

Let's look at the ``train`` method first::

    def train(self, train_data: TrainData, train_config: TrainConfig) -> TrainResult:
        ...

This method takes in the training data and training config (i.e., the number of epochs to train for, the batch size, etc.), trains the model, and return some training results (i.e., average loss, accuracy, etc.).
The ``TrainData`` class is wrapper class for storing the training data and can be iterated over to get the input features/target values for each training example.
The ``TrainConfig`` and ``TrainResult`` classes are essentially dictionaries that store training configuration parameter values and training results, respectively.

The ``test`` method is similar to the ``train`` method::

    def test(self, test_data: TestData, test_config: TestConfig) -> tuple[dict[str, Target], TestResult]:
        ...

This method takes in the test data and test config (i.e., the batch size, etc.), tests the model, and returns the predictions and some test results (i.e., average loss, accuracy, etc.).
The ``TestData``, ``TestConfig``, and ``TestResult`` classes function in a similar manner to their training counterparts.

Now, given that you will provide the input data on your own, there is one final step to creating the CLI for your Kaggle Competition model development.
You need to create an instance of your custom dataset class and pass it to a function that will initialize the CLI::

    dataset = MyDataset()
    if __name__ == "__main__":
        kd.run(dataset)

Now, you can run your CLI with the following command::

    python3 my_script.py -h

This will print out the help message for the CLI.
When training/testing a model, you will need to pass the name of the model architecture from earlier and a configuration file that contains the hyperparameter values to use.
The fields in the configuration file must match the names of the parameters in the model class.
For example, if you wanted to train a model with the name `my_model` and the hyperparameter values ``param1 = 1`` and ``param2 = 0.5``, you would run the following command::

    python3 my_script.py train my_model --model_config_file config.yml

where ``config.yml`` contains the following::

    param1: 1
    param2: 0.5

Now, this is a very high-level overview of how to create a CLI for your Kaggle Competition model development.
Many details were left out for brevity.
If you would like to see complete examples of how to use the Kaggle Driver to create a CLI for your Kaggle Competition model development, see :ref:`Examples<>`.
