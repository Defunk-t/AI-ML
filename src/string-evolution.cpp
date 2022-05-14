#include <cstdlib>
#include <ctime>

#include "ai/ga/StringEvolver.cpp"

int main()
{
	std::srand(std::time(nullptr));
	new GA::StringEvolver(250, 40, 5, 0.04, "string");
	return 0;
};
