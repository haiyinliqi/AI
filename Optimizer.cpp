#include"Optimizer.h"
namespace AI {
	Optimizer::_One_ai_group::_One_parameter_matrix::_One_parameter_matrix(parameter_t** parameters, parameter_t** def, int height, int width) noexcept :
		_parameters(parameters), _def(def), _height(height), _width(width) {
	}
	void Optimizer::_One_ai_group::del() noexcept {
	}
	void Optimizer::use_strategy(parameter_t parameter, parameter_t& def) noexcept {
		for (Optimization_strategy* Optimization_strategy : _optimizationStrategies) {
			Optimization_strategy->use_strategy(parameter, def);
		}
	}
	void Optimizer::add_strategy(Optimization_strategy* Optimization_strategy) noexcept {
		_optimizationStrategies.insert(std::lower_bound(_optimizationStrategies.begin(), _optimizationStrategies.end(), Optimization_strategy), Optimization_strategy);
	}
	SGD::_SGD_one_ai_group::_SGD_one_parameter_matrix::_SGD_one_parameter_matrix(parameter_t** parameters, parameter_t** def, int height, int width) noexcept :
		_One_parameter_matrix(parameters, def, height, width) {
	}
	int SGD::init() noexcept {
		_all_ai.emplace_back();
		_max_id++;

		return _max_id;
	}
	void SGD::file_parameters(int id, parameter_t** parameters, parameter_t** def, int height, int width) noexcept {
		_all_ai[id]._parameter_matrices.emplace_back(parameters, def, height, width);
	}
	void SGD::register_parameters(int id, parameter_t lr) noexcept {
		for (_SGD_one_ai_group::_SGD_one_parameter_matrix& i : _all_ai[id]._parameter_matrices) {
			for (int j = 0;j < i._height;j++) {
				for (int l = 0;l < i._width;l++) {
					use_strategy(i._parameters[j][l], i._def[j][l]);
					i._parameters[j][l] -= i._def[j][l] * lr;
					i._def[j][l] = 0;
				}
			}
		}
	}
	void SGD::del(int id) noexcept {
		_all_ai[id].del();
	}
	Adam::_Adam_one_ai_group::_Adam_one_parameter_matrix::_Adam_one_parameter_matrix(parameter_t** parameters, parameter_t** def, int height, int width) noexcept :
		_One_parameter_matrix(parameters, def, height, width) {
		_m = new parameter_t * [height];
		_v = new parameter_t * [height];
		for (int i = 0;i < height;i++) {
			_m[i] = new parameter_t[width];
			_v[i] = new parameter_t[width];
			for (int j = 0;j < width;j++) {
				_m[i][j] = 0;
				_v[i][j] = 0;
			}
		}
	}
	void Adam::_Adam_one_ai_group::del() noexcept {
		for (_Adam_one_parameter_matrix i : _parameter_matrices) {
			for (int j = 0;j < i._height;j++) {
				delete[] i._m[j];
				delete[] i._v[j];
			}
			delete[] i._m;
			delete[] i._v;
		}
		is_del = true;
	}
	int Adam::init() noexcept {
		_all_ai.emplace_back();
		_max_id++;

		return _max_id;
	}
	void Adam::file_parameters(int id, parameter_t** parameters, parameter_t** def, int height, int width) noexcept {
		_all_ai[id]._parameter_matrices.emplace_back(parameters, def, height, width);
	}
	void Adam::register_parameters(int id, parameter_t lr) noexcept {
		_all_ai[id]._t++;
		for (_Adam_one_ai_group::_Adam_one_parameter_matrix& i : _all_ai[id]._parameter_matrices) {
			for (int j = 0;j < i._height;j++) {
				for (int l = 0;l < i._width;l++) {
					use_strategy(i._parameters[j][l], i._def[j][l]);
					i._m[j][l] *= _beta1;
					i._m[j][l] += (1 - _beta1) * i._def[j][l];
					i._v[j][l] *= _beta2;
					i._v[j][l] += (1 - _beta2) * std::pow(i._def[j][l], 2);
					i._parameters[j][l] -= i._m[j][l] / (1 - pow(_beta1, _all_ai[id]._t)) / (std::sqrt(i._v[j][l] / (1 - pow(_beta2, _all_ai[id]._t))) + _epsilon) * lr;
					i._parameters[j][l] -= i._def[j][l] * lr;
					i._def[j][l] = 0;
				}
			}
		}
	}
	void Adam::del(int id) noexcept {
		_all_ai[id].del();
	}
	Adam::Adam(parameter_t beta1, parameter_t beta2, parameter_t epsilon) noexcept :
		_beta1(beta1), _beta2(beta2), _epsilon(epsilon) {
	}
	Adam::~Adam() noexcept {
		for (int i = 0;i < _all_ai.size();i++) {
			if (_all_ai[i].is_del) {
				continue;
			}
			_all_ai[i].del();
		}
	}
}