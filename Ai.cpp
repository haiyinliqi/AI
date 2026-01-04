#include"Ai.h"
namespace AI {
	Ai::Ai(int size, Layer** layers, LossFunction_kinds lossfuncion_type, Optimizer* optimizer, parameter_t** output, parameter_t** def, int input_num, int output_num) noexcept :
		_size(size), _layers(layers), _optimizer(optimizer), _cnt(0), _samples(nullptr), _output(output), _def(def), _input_num(input_num), _output_num(output_num) {
		_lossFunction = LossFunctionFactory::create_lossFunction(lossfuncion_type, _output[_size], _def[_size], output_num);
		_optimizer_id = _optimizer->init();
		file_parameters();
	}
	void Ai::file_parameters() noexcept {
		for (int i = 0;i < _size;i++) {
			_layers[i]->file_parameters(_optimizer, _optimizer_id);
		}
	}
	void Ai::set_samples(std::string file, int cnt, bool have_expectation) noexcept {
		for (int i = 0;i < _cnt;i++) {
			delete[] _samples[i]._input;
			delete[] _samples[i]._expectation;
		}
		delete[] _samples;
		_cnt = cnt;
		_samples = new _Sample[cnt];
		std::ifstream fin(file);
		for (int i = 0;i < _cnt;i++) {
			_samples[i]._input = new parameter_t[_input_num];
			_samples[i]._expectation = new parameter_t[_output_num];
			for (int j = 0;j < _input_num;j++) {
				fin >> _samples[i]._input[j];
			}
			for (int j = 0;j < _output_num && have_expectation;j++) {
				fin >> _samples[i]._expectation[j];
			}
		}
	}
	void Ai::set_parameters(const Dynamic_tensor& parameters) noexcept {
		for (int i = 0;i < _size;i++) {
			_layers[i]->set_parameters(parameters[i]);
		}
	}
	void Ai::set_optimizer(Optimizer* optimizer) noexcept {
		_optimizer->del(_optimizer_id);
		_optimizer = optimizer;
		_optimizer_id = _optimizer->init();
		file_parameters();
	}
	parameter_t* Ai::compute(parameter_t* input) const noexcept {
		for (int i = 0;i < _input_num;i++) {
			_output[0][i] = input[i];
		}
		for (int i = 0;i < _size;i++) {
			_layers[i]->calculate_output();
		}
		parameter_t* ans = new parameter_t[_output_num];
		for (int i = 0;i < _output_num;i++) {
			ans[i] = _output[_size][i];
		}

		return ans;
	}
	parameter_t Ai::register_parameters(parameter_t* expectation, parameter_t lr) noexcept {
		parameter_t ans = _lossFunction->lossFunction(expectation);
		for (int i = _size;i > 0;i--) {
			_layers[i - 1]->register_parameters();
		}
		_optimizer->register_parameters(_optimizer_id, lr);

		return ans;
	}
	void Ai::train(int n, parameter_t lr) noexcept {
		for (int i = 0;i < n;i++) {
			parameter_t loss = 0;
			std::shuffle(_samples, _samples + _cnt, gen);
			for (int j = 0;j < _cnt;j++) {
				for (int l = 0;l < _input_num;l++) {
					_output[0][l] = _samples[j]._input[l];
				}
				for (int l = 0;l < _size;l++) {
					_layers[l]->calculate_output();
				}
				loss += register_parameters(_samples[j]._expectation, lr);
			}
			loss /= _cnt;
		}
	}
	void Ai::train(parameter_t expect, int maxn, parameter_t lr) noexcept {
		for (int i = 0;i < maxn;i++) {
			parameter_t loss = 0;
			std::shuffle(_samples, _samples + _cnt, gen);
			for (int j = 0;j < _cnt;j++) {
				for (int l = 0;l < _input_num;l++) {
					_output[0][l] = _samples[j]._input[l];
				}
				for (int l = 0;l < _size;l++) {
					_layers[l]->calculate_output();
				}
				loss += register_parameters(_samples[j]._expectation, lr);
			}
			loss /= _cnt;
			std::system("cls");
			std::cout << i << std::endl << loss;
			if (loss <= expect) {

				return;
			}
		}
	}
	Dynamic_tensor Ai::get_parameters() const noexcept {
		Dynamic_tensor ans;
		ans.resize(_size);
		for (int i = 0;i < _size;i++) {
			ans[i] = std::move(_layers[i]->get_parameters());
		}

		return ans;
	}
	Ai::~Ai() noexcept {
		_optimizer->del(_optimizer_id);
		for (int i = 0;i < _size;i++) {
			delete _layers[i];
		}
		for (int i = 0;i < _cnt;i++) {
			delete[] _samples[i]._input;
			delete[] _samples[i]._expectation;
		}
		delete[] _samples;
		delete[] _output[0];
		delete[] _def[0];
		delete _lossFunction;
	}
	Compute_ai::Compute_ai(int size, Layer** layers, parameter_t** output, int input_num, int output_num) noexcept :
		_size(size), _layers(layers), _output(output), _input_num(input_num), _output_num(output_num) {
	}
	void Compute_ai::set_parameters(const Dynamic_tensor& parameters) noexcept {
		for (int i = 0;i < _size;i++) {
			_layers[i]->set_parameters(parameters[i]);
		}
	}
	parameter_t* Compute_ai::compute(parameter_t* input) const noexcept {
		for (int i = 0;i < _input_num;i++) {
			_output[0][i] = input[i];
		}
		for (int i = 0;i < _size;i++) {
			_layers[i]->calculate_output();
		}
		parameter_t* ans = new parameter_t[_output_num];
		for (int i = 0;i < _output_num;i++) {
			ans[i] = _output[_size][i];
		}

		return ans;
	}
	const Dynamic_tensor& Compute_ai::get_parameters() const noexcept {
		Dynamic_tensor* ans = new Dynamic_tensor;
		(*ans).resize(_size);
		for (int i = 0;i < _size;i++) {
			(*ans)[i] = _layers[i]->get_parameters();
		}

		return (*ans);
	}
	Compute_ai::~Compute_ai() noexcept {
		for (int i = 0;i < _size;i++) {
			delete _layers[i];
		}
		delete[] _output[0];
	}
	Ai* Ai_builder::_Set_lossFunction::set_lossFunction(LossFunction_kinds lossFunction_type) noexcept {
		Ai* ai = new Ai(static_cast<int>(Ai_builder::_layers.size()), Ai_builder::_layers.data(), lossFunction_type, Ai_builder::_optimizer, Ai_builder::_output.data(), Ai_builder::_def.data(), Ai_builder::_ai_input_num, Ai_builder::_input_num);
		Ai_builder::_layers.clear();
		Ai_builder::_output.clear();
		Ai_builder::_def.clear();

		return ai;
	}
	Ai_builder::_Set_lossFunction* Ai_builder::_Add_layer::set_optimizer(Optimizer* optimizer) noexcept {
		Ai_builder::_optimizer = optimizer;

		return nullptr;
	}
	Ai_builder::_Add_layer* Ai_builder::_Set_input::set_input_num(int input_num) noexcept {
		Ai_builder::_ai_input_num = Ai_builder::_input_num = input_num;
		Ai_builder::_output.push_back(new parameter_t[_input_num]);
		Ai_builder::_def.push_back(new parameter_t[_input_num]);

		return nullptr;
	}
	Ai_builder::_Set_input* Ai_builder::init() noexcept {
		for (Layer*& i : _layers) {
			delete i;
		}
		if (_output.size() != 0) {
			delete[] _output[0];
			delete[] _def[0];
		}
		_layers.clear();
		_output.clear();
		_optimizer = nullptr;
		_ai_input_num = _input_num = 0;

		return nullptr;
	}
}