#include"ActivationFunction.h"
namespace AI {
	ActivationFunction::ActivationFunction(int output_num, parameter_t* output, parameter_t* def) noexcept :
		_output_num(output_num), _output(output), _def(def) {
	}
	void Null::compute_output() noexcept {
	}
	void Null::compute_def() noexcept {
	}
	void Relu::compute_output() noexcept {
		for (int i = 0;i < _output_num;i++) {
			if (_output[i] < 0) {
				_output[i] = 0;
			}
		}
	}
	void Relu::compute_def() noexcept {
		for (int i = 0;i < _output_num;i++) {
			if (_output[i] == 0) {
				_def[i] = 0;
			}
		}
	}
	void LeakyRelu::compute_output() noexcept {
		for (int i = 0;i < _output_num;i++) {
			if (_output[i] < 0) {
				_output[i] *= 1e-2;
			}
		}
	}
	void LeakyRelu::compute_def() noexcept {
		for (int i = 0;i < _output_num;i++) {
			if (_output[i] <= 0) {
				_def[i] *= 1e-2;
			}
		}
	}
	void Tanh::compute_output() noexcept {
		for (int i = 0;i < _output_num;i++) {
			_output[i] = std::tanh(_output[i]);
		}
	}
	void Tanh::compute_def() noexcept {
		for (int i = 0;i < _output_num;i++) {
			_def[i] *= 1 - _output[i] * _output[i];
		}
	}
	Softmax::Softmax(int output_num, parameter_t* output, parameter_t* def) noexcept :
		ActivationFunction(output_num, output, def) {
		_def_sum = new parameter_t[output_num];
	}
	void Softmax::compute_output() noexcept {
		parameter_t sum = 0, max_input = 0;
		for (int i = 0;i < _output_num;i++) {
			max_input = std::max(max_input, _output[i]);
		}
		for (int i = 0;i < _output_num;i++) {
			_output[i] = std::exp(_output[i] - max_input);
			sum += _output[i];
		}
		for (int i = 0;i < _output_num;i++) {
			_output[i] = _output[i] / sum;
		}
	}
	void Softmax::compute_def() noexcept {
		for (int i = 0;i < _output_num;i++) {
			_def_sum[i] = 0;
			for (int j = 0;j < _output_num;j++) {
				if (i == j) {
					_def_sum[i] += _output[j] * (1 - _output[j]) * _def[j];
				}
				else {
					_def_sum[i] += -_output[i] * _output[j] * _def[j];
				}
			}
		}
		for (int i = 0;i < _output_num;i++) {
			_def[i] = _def_sum[i];
		}
	}
	Softmax::~Softmax() noexcept {
		delete[] _def_sum;
	}
}
