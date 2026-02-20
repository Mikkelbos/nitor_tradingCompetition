from process_data import setup
from reg_estimate import run_regression
from evaluate_model import evaluatep


def run_model():

    data, val_data = setup()

    model = run_regression(data)

    submit = evaluate(data,val_data, model)

    print(f"Script ran succesfully")
if __name__ == '__main__':
    run_model()














