#include"Layer.h"
#include<iostream>
namespace AI {
	std::random_device rd;
	std::mt19937 gen(rd() ^ time(nullptr));
	std::uniform_real_distribution<parameter_t> dis(-1, 1);
	Layer::Layer(int input_num, int output_num, parameter_t* input, parameter_t*& output, parameter_t* input_def, parameter_t*& output_def, bool only_compute) noexcept :
		_input_num(input_num), _output_num(output_num), _input(input), _input_def(input_def), _output(output), _output_def(output_def), _only_compute(only_compute) {
		_output = output = new parameter_t[_output_num];
		if (!_only_compute) {
			_output_def = output_def = new parameter_t[_output_num];
		}
	}
	void Layer::file_parameters(parameter_t** parameters, parameter_t** def, int height, int width, Optimizer* optimizer, int optimizer_id) noexcept {
		optimizer->file_parameters(optimizer_id, parameters, def, height, width);
	}
	Layer::~Layer() noexcept {
		delete[] _output;
		delete[] _output_def;
	}
}