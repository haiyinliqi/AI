#ifndef OPTIMIZATION_STRATEGY_H_
#define OPTIMIZATION_STRATEGY_H_
#include"ai_type.h"
namespace AI {
	class Optimization_strategy {//优化策略类
	protected:
		//使用策略
		virtual void use_strategy(parameter_t parameter, parameter_t& def) noexcept = 0;
		//获取排序权值
		virtual int get_order() const noexcept = 0;
	public:
		//按权值比较（梯度剪枝排在最后，防止正则化后梯度超出阈值）
		bool operator<(const Optimization_strategy& other) const noexcept;
		friend class Optimizer;
	};
	class L1Regularization :public Optimization_strategy {//L1正则化类
	protected:
		parameter_t _gamma;//正则化系数
		void use_strategy(parameter_t parameter, parameter_t& def) noexcept override;
		int get_order() const noexcept override;
	public:
		L1Regularization(parameter_t gamma = 1e-1) noexcept;
	};
	class L2Regularization :public Optimization_strategy {//L2正则化类
	protected:
		parameter_t _gamma;//正则化系数
		void use_strategy(parameter_t parameter, parameter_t& def) noexcept override;
		int get_order() const noexcept override;
	public:
		L2Regularization(parameter_t gamma = 1e-2) noexcept;
	};
	class GradientClipping :public Optimization_strategy {//梯度剪枝类
	protected:
		parameter_t _max_def;//导数最大值（最大不超过_max_def，最小不低于_max_def的倒数）
		void use_strategy(parameter_t parameter, parameter_t& def) noexcept override;
		int get_order() const noexcept override;
	public:
		GradientClipping(parameter_t max_def = 1) noexcept;
	};
}
#endif