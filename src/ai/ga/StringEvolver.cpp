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
};
