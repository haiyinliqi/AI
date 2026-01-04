#ifndef _ACTIVATIONFUNCTION_H_
#define _ACTIVATIONFUNCTION_H_ true
#include<map>
#include<cmath>
#include"ai_type.h"
namespace AI {
	class ActivationFunction {//激活函数类
	protected:
		parameter_t* _output, * _def;//输出、导数
		int _output_num;//上一层神经元数量、本层神经元数量
	public:
		ActivationFunction(int output_num, parameter_t* output, parameter_t* def) noexcept;
		virtual ~ActivationFunction() noexcept = default;
		friend class Layer;
	};
	class Null :public ActivationFunction {//空激活函数
	public:
		using ActivationFunction::ActivationFunction;
		//计算输出
		void compute_output() noexcept;
		//计算导数
		void compute_def() noexcept;
		friend class ActivationFunctionFactory;
	};
	class Relu :public ActivationFunction {//Relu激活函数
	public:
		using ActivationFunction::ActivationFunction;
		//计算输出
		void compute_output() noexcept;
		//计算导数
		void compute_def() noexcept;
		friend class ActivationFunctionFactory;
	};
	class LeakyRelu :public ActivationFunction {//LeakyRelu激活函数
	public:
		using ActivationFunction::ActivationFunction;
		//计算输出
		void compute_output() noexcept;
		//计算导数
		void compute_def() noexcept;
		friend class ActivationFunctionFactory;
	};
	class Tanh :public ActivationFunction {//Tanh激活函数
	public:
		using ActivationFunction::ActivationFunction;
		//计算输出
		void compute_output() noexcept;
		//计算导数
		void compute_def() noexcept;
		friend class ActivationFunctionFactory;
	};
	class Softmax :public ActivationFunction {//Softmax激活函数
	protected:
		parameter_t* _def_sum;//导数和
	public:
		Softmax(int output_num, parameter_t* output, parameter_t* def) noexcept;
		//计算输出
		void compute_output() noexcept;
		//计算导数
		void compute_def() noexcept;
		~Softmax() noexcept;
		friend class ActivationFunctionFactory;
	};
}
#endif

