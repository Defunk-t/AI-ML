#include <iostream>
#include <cstdlib>

namespace GA
{
	typedef unsigned short GA_Int;

	template <class T>
	class GeneticAlgorithm
	{
	  public:
		void run();

	  private:
		static const GA_Int ARRAY_SIZE = 65535;

	  protected:
		GA_Int populationSize{};
		GA_Int genePoolSize{};
		GA_Int eliteCount{};
		double mutationChance{};
		unsigned long generationCount = 1;
		T individuals[ARRAY_SIZE]{};
		int fitnesses[ARRAY_SIZE]{};
		GA_Int sortedIndexes[ARRAY_SIZE]{};
		T genePool[ARRAY_SIZE]{};

		virtual T createIndividual() const = 0;
		virtual T createIndividual(T parentA, T parentB) const = 0;
		virtual int testFitness(T individual) const = 0;
		virtual bool terminateCondition(int) const { return false; };
		virtual void print() const {};

		T rollForParent(T a, T b) const { return std::rand() % 2 == 0 ? a : b; };
		bool rollForMutation() const { return std::rand() % 10000 / 10000 < mutationChance; };
		int getCurrentBestFitness() const { return fitnesses[sortedIndexes[0]]; };

		GeneticAlgorithm(
			GA_Int populationSize,
			GA_Int genePoolSize,
			GA_Int eliteCount,
			double mutationChance
		);

	};
};
