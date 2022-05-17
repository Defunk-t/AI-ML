#include "Eggholder.h"

namespace GA
{
	Eggholder::Eggholder(GAIndex bits, GAIndex populationSize)
	{
		if (bits * populationSize > MAX_ARRAY) throw std::exception();
		this->bits = bits;
		this->populationSize = populationSize;
		generatePopulation();
		printPopulation();
	}
	GAIndex Eggholder::getPos(GAIndex individual, GAIndex bit) const
	{
		return bits * individual + bit;
	}
	void Eggholder::generatePopulation()
	{
		for (GAIndex i = 0; i < populationSize; i++)
			for (GAIndex j = 0; j < bits; j++)
				generation[getPos(i, j)] = std::rand() % 2 == 1 ? true : false;
	}
	void Eggholder::printPopulation()
	{
		for (GAIndex i = 0; i < populationSize; i++)
		{
			for (GAIndex j = 0; j < bits; j++)
				std::cout << generation[getPos(i, j)];
			std::cout << std::endl;
		}
	}
}
