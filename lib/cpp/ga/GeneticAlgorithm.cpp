#include "GeneticAlgorithm.h"

namespace GA
{
	template <class T>
	GeneticAlgorithm<T>::GeneticAlgorithm(
		GA_Int populationSize,
		GA_Int genePoolSize,
		GA_Int eliteCount,
		double mutationChance)
	{
		this->populationSize = populationSize;
		this->genePoolSize = genePoolSize;
		this->eliteCount = eliteCount;
		this->mutationChance = mutationChance;
	};

	template <class T>
	void GeneticAlgorithm<T>::run()
	{

		// Initial population
		for (int i = 0; i < populationSize; i++)
		{
			fitnesses[i] = testFitness(individuals[i] = createIndividual());
			sortedIndexes[i] = i;
		}

		while (!terminateCondition(fitnesses[0]))
		{
			// Sort
			for (int i = 0; i < populationSize; i++)
				for (int j = 0, k = 1; k < populationSize - i; j++, k++)
					if (fitnesses[sortedIndexes[j]] > fitnesses[sortedIndexes[k]]) {
						GA_Int temp = sortedIndexes[j];
						sortedIndexes[j] = sortedIndexes[k];
						sortedIndexes[k] = temp;
					}

			for (int i = 0; i < genePoolSize; i++)
				genePool[i] = individuals[sortedIndexes[i]];

			print();

			for (int i = 0; i < eliteCount; i++)
			{
				individuals[i] = genePool[i];
				fitnesses[i] = fitnesses[sortedIndexes[i]];
				sortedIndexes[i] = i;
			}

			for (int i = eliteCount; i < populationSize; i++)
			{
				GA_Int parentA = rand() % genePoolSize, parentB;
				do parentB = rand() % genePoolSize;
				while (parentA == parentB);

				fitnesses[i] = testFitness(individuals[i] = createIndividual(genePool[parentA], genePool[parentB]));
				sortedIndexes[i] = i;
			}

			generationCount++;
		}
	};
};
