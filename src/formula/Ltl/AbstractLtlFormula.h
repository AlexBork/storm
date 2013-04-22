/*
 * AbstractLtlFormula.h
 *
 *  Created on: 15.04.2013
 *      Author: thomas
 */

#ifndef STORM_LTL_ABSTRACTLTLFORMULA_H_
#define STORM_LTL_ABSTRACTLTLFORMULA_H_

#include <vector>
#include "src/modelchecker/ForwardDeclarations.h"
#include "src/formula/abstract/AbstractFormula.h"

namespace storm {
namespace formula {
namespace ltl {

template <class T>
class AbstractLtlFormula : public virtual storm::formula::abstract::AbstractFormula<T> {
public:
	/**
	 * Empty destructor
	 */
	virtual ~AbstractLtlFormula() {
		// Intentionally left empty
	}

	/*!
	 * Calls the model checker to check this formula.
	 * Needed to infer the correct type of formula class.
	 *
	 * @note This function should only be called in a generic check function of a model checker class. For other uses,
	 *       the methods of the model checker should be used.
	 *
	 * @note This function is not implemented in this class.
	 *
	 * @returns A vector indicating the probability that the formula holds for each state.
	 */
	virtual std::vector<T>* check(const storm::modelchecker::AbstractModelChecker<T>& modelChecker) const = 0;

	/*!
	 * Clones the called object.
	 *
	 * Performs a "deep copy", i.e. the subtrees of the new object are clones of the original ones
	 *
	 * @returns a new AND-object that is identical the called object.
	 */
	virtual AbstractLtlFormula<T>* clone() const = 0;
};

} /* namespace ltl */
} /* namespace formula */
} /* namespace storm */
#endif /* STORM_LTL_ABSTRACTLTLFORMULA_H_ */
