#include"LossFunction.h"
namespace AI {
	LossFunction::LossFunction(parameter_t* output, parameter_t* def, int output_num) noexcept :
		_output(output), _def(def), _output_num(output_num) {
	}
	LossFunction* LossFunctionFactory::create_lossFunction(LossFunction_kinds key, parameter_t* output, parameter_t* def, int output_num) noexcept {

		return _index[static_cast<int>(key)](output, def, output_num);
	}
	parameter_t MeanSquaredError::lossFunction(parameter_t* expectation) noexcept {
		parameter_t sum = 0;
		for (int i = 0;i < _output_num;i++) {
			sum += std::pow(_output[i] - expectation[i], 2);
			_def[i] = (_output[i] - expectation[i]) * 2 / _output_num;
		}

		return sum;
	}
	parameter_t CrossEntropyLoss::lossFunction(parameter_t* expectation) noexcept {
		parameter_t sum = 0;
		for (int i = 0;i < _output_num;i++) {
			sum += -expectation[i] * std::log(_output[i]);
			_def[i] = -expectation[i] / _output[i];
		}

		return sum;
	}
}
