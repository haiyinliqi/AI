#ifndef LOSSFUNCTION_H_
#define LOSSFUNCTION_H_
#include<map>
#include<cmath>
#include"ai_type.h"
namespace AI {
	class LossFunction {//损失函数类
	protected:
		LossFunction(parameter_t* output, parameter_t* def, int num) noexcept;
		parameter_t* _output, * _def;//输出、导数
		int _output_num;//神经网络输出个数、层数
		//计算损失
		virtual parameter_t lossFunction(parameter_t* expectation) noexcept = 0;
		virtual ~LossFunction() noexcept = default;
		friend class Ai;
	};
	enum class LossFunction_kinds {//损失函数类型枚举类
		MeanSquaredError, CrossEntropyLoss
	};
	class LossFunctionFactory {//损失函数工厂
	protected:
		//损失函数构造索引表
		inline static std::map<int, LossFunction* (*)(parameter_t*, parameter_t*, int)> _index;
		LossFunctionFactory() = delete;
		//构造损失函数派生类
		static LossFunction* create_lossFunction(LossFunction_kinds key, parameter_t* output, parameter_t* def, int num) noexcept;
		friend class Ai;
	public:
		//注册损失函数
		template<class LossFunction_name>
		static void register_lossFunction(int key) noexcept {
			_index[key] = [](parameter_t* output, parameter_t* def, int output_num)->LossFunction* {

				return new LossFunction_name(output, def, output_num);
				};
		}
	};
	//损失函数派生类宏
#define LossFunctionDerived(name) class name :public LossFunction {\
	private:\
		struct Registant {\
			Registant() {\
				LossFunctionFactory::register_lossFunction<name>(static_cast<int>(LossFunction_kinds::name));\
			}\
		};\
		inline static Registant _registant;\
	protected:\
		parameter_t lossFunction(parameter_t* expectation) noexcept override;\
		using LossFunction::LossFunction;\
		friend class LossFunctionFactory;\
	};
	LossFunctionDerived(MeanSquaredError);//均方损失函数
	LossFunctionDerived(CrossEntropyLoss);//交叉熵损失函数
}
#endif
