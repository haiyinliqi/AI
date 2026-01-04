#ifndef _DYNAMIC_TENSOR_H_
#define _DYNAMIC_TENSOR_H_ true
#include<vector>
#include<iostream>
#include"ai_type.h"
namespace AI {
	class Dynamic_tensor {//可变矩阵类
	protected:
		parameter_t _value = 0;//当前节点所存储的值
		std::vector<Dynamic_tensor> _matrices;//下一级子节点
	public:
		Dynamic_tensor() = default;
		Dynamic_tensor(const Dynamic_tensor& other) = default;
		//隐式转换为long double
		operator parameter_t() const noexcept;
		//直接使用long double赋值
		parameter_t& operator=(const parameter_t& value) noexcept;
		//移动赋值
		void operator =(Dynamic_tensor&& other) noexcept;
		//改变长度（不减）
		void resize(const int& size) noexcept;
		//使用下标访问下一级元素
		Dynamic_tensor& operator[](const int& index) noexcept;
		//使用下标访问下一级元素
		const Dynamic_tensor& operator[](const int& index) const noexcept;
		//重载输出
		friend std::ostream& operator<<(std::ostream& os, const Dynamic_tensor& matrix);
		//重载输入
		friend std::istream& operator>>(std::istream& is, Dynamic_tensor& matrix);
	};
}
#endif