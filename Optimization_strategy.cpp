#include"Optimization_strategy.h"
namespace AI {
	bool Optimization_strategy::operator<(const Optimization_strategy& other) const noexcept {

		return get_order() < other.get_order();
	}
	void L1Regularization::use_strategy(parameter_t parameter, parameter_t& def) noexcept {
		def += (parameter < 0 ? -1 : 1) * _gamma;
	}
	int L1Regularization::get_order() const noexcept {

		return 1;
	}
	L1Regularization::L1Regularization(parameter_t gamma) noexcept :_gamma(gamma) {
	}
	void L2Regularization::use_strategy(parameter_t parameter, parameter_t& def) noexcept {
		def += parameter * 2 * _gamma;
	}
	int L2Regularization::get_order() const noexcept {

		return 1;
	}
	L2Regularization::L2Regularization(parameter_t gamma) noexcept :_gamma(gamma) {
	}
	void GradientClipping::use_strategy(parameter_t parameter, parameter_t& def) noexcept {
		def = (def > _max_def ? _max_def : def);
		def = (def < -_max_def ? -_max_def : def);
	}
	int GradientClipping::get_order() const noexcept {

		return 2;
	}
	GradientClipping::GradientClipping(parameter_t max_def) noexcept :_max_def(max_def) {
	}
}