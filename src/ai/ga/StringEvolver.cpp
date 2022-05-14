#include "StringEvolver.h"

namespace GA
{
	StringEvolver::StringEvolver(
		GA_Int populationSize,
		GA_Int genePoolSize,
		GA_Int eliteCount,
		double mutationChance,
		const string &targetString
	) : GeneticAlgorithm<array<char,256>>(populationSize, genePoolSize, eliteCount, mutationChance)
	{
		this->targetString = targetString;
		this->targetStringSize = (StringIndex) targetString.size();
		run();
	};

	char StringEvolver::getRandomChar()
	{
		return (char) (32 + rand() % 95);
	};

	String StringEvolver::createIndividual() const
	{
		String s;
		for (StringIndex i = 0; i < targetStringSize; i++)
			s[i] = getRandomChar();
		return s;
	};

	String StringEvolver::createIndividual(String parentA, String parentB) const
	{
		String s;
		for (StringIndex i = 0; i < targetStringSize; i++)
		{
			if (rollForMutation()) s[i] = getRandomChar();
			else s[i] = rollForParent(parentA, parentB)[i];
		}
		return s;
	};

	int StringEvolver::testFitness(String s) const
	{
		int fitness = 0;
		for (StringIndex i = 0; i < targetStringSize; i++)
			if (s[i] != targetString[i]) ++fitness;
		return fitness;
	};

	bool StringEvolver::terminateCondition(int fitness) const
	{
		return fitness == 0;
	};

	void StringEvolver::print() const
	{
		if (generationCount % 10000 == 0)
		{
			cout << "Generation: " << generationCount << endl << "Best fitness: " << getCurrentBestFitness() << endl;

			for (int i = 0; i < 5 && i < genePoolSize; i++)
			{
				for (StringIndex j = 0; j < targetStringSize; j++)
					cout << genePool[i][j];
				cout << "\t";
			}

			cout << endl << endl;
		}
	};
};
