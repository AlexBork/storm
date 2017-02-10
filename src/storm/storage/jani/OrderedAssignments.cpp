#include "storm/storage/jani/OrderedAssignments.h"

#include "storm/utility/macros.h"
#include "storm/exceptions/InvalidArgumentException.h"
#include "storm/storage/jani/VariableSet.h"

namespace storm {
    namespace jani {
        
        OrderedAssignments::OrderedAssignments(std::vector<Assignment> const& assignments) {
            for (auto const& assignment : assignments) {
                add(assignment);
            }
        }
        
        OrderedAssignments::OrderedAssignments(Assignment const& assignment) {
            add(assignment);
        }
        
        bool OrderedAssignments::add(Assignment const& assignment) {
            // If the element is contained in this set of assignment, nothing needs to be added.
            if (this->contains(assignment)) {
                return false;
            }
            
            // Otherwise, we find the spot to insert it.
            auto it = lowerBound(assignment, allAssignments);

            if (it != allAssignments.end()) {
                STORM_LOG_THROW(assignment.getExpressionVariable() != (*it)->getExpressionVariable(), storm::exceptions::InvalidArgumentException, "Cannot add assignment ('" << assignment.getAssignedExpression() << "') as an assignment ('" << (*it)->getAssignedExpression()  << "') to variable '" <<  (*it)->getVariable().getName() << "' already exists.");
            }
            
            // Finally, insert the new element in the correct vectors.
            auto elementToInsert = std::make_shared<Assignment>(assignment);
            allAssignments.emplace(it, elementToInsert);
            if (assignment.isTransient()) {
                auto transientIt = lowerBound(assignment, transientAssignments);
                transientAssignments.emplace(transientIt, elementToInsert);
            } else {
                auto nonTransientIt = lowerBound(assignment, nonTransientAssignments);
                nonTransientAssignments.emplace(nonTransientIt, elementToInsert);
            }
            
            return true;
        }
        
        bool OrderedAssignments::remove(Assignment const& assignment) {
            // If the element is contained in this set of assignment, nothing needs to be removed.
            if (!this->contains(assignment)) {
                return false;
            }
            
            // Otherwise, we find the element to delete.
            auto it = lowerBound(assignment, allAssignments);
            STORM_LOG_ASSERT(it != allAssignments.end(), "Invalid iterator, expected existing element.");
            STORM_LOG_ASSERT(assignment == **it, "Wrong iterator position.");
            allAssignments.erase(it);
            
            if (assignment.isTransient()) {
                auto transientIt = lowerBound(assignment, transientAssignments);
                STORM_LOG_ASSERT(transientIt != transientAssignments.end(), "Invalid iterator, expected existing element.");
                STORM_LOG_ASSERT(assignment == **transientIt, "Wrong iterator position.");
                transientAssignments.erase(transientIt);
            } else {
                auto nonTransientIt = lowerBound(assignment, nonTransientAssignments);
                STORM_LOG_ASSERT(nonTransientIt != nonTransientAssignments.end(), "Invalid iterator, expected existing element.");
                STORM_LOG_ASSERT(assignment == **nonTransientIt, "Wrong iterator position.");
                nonTransientAssignments.erase(nonTransientIt);
            }
            return true;
        }
        
        bool OrderedAssignments::hasMultipleLevels() const {
            if (allAssignments.empty()) {
                return false;
            }
            return getLowestLevel() != getHighestLevel();
        }
        
        bool OrderedAssignments::empty() const {
            return allAssignments.empty();
        }
        
        void OrderedAssignments::clear() {
            allAssignments.clear();
            transientAssignments.clear();
            nonTransientAssignments.clear();
        }
        
        std::size_t OrderedAssignments::getNumberOfAssignments() const {
            return allAssignments.size();
        }
        
        int_fast64_t OrderedAssignments::getLowestLevel() const {
            assert(!allAssignments.empty());
            return allAssignments.front()->getLevel();
        }
        
        int_fast64_t OrderedAssignments::getHighestLevel() const {
            assert(!allAssignments.empty());
            return allAssignments.back()->getLevel();
        }
        
        bool OrderedAssignments::contains(Assignment const& assignment) const {
            auto it = lowerBound(assignment, allAssignments);
            if (it != allAssignments.end() && assignment == **it) {
                return true;
            } else {
                return false;
            }
        }

        OrderedAssignments OrderedAssignments::simplifyLevels(bool synchronous, VariableSet const& localVars) const {
            bool changed = false;
            std::vector<Assignment> newAssignments;
            for (auto const& assignment : allAssignments) {
                newAssignments.push_back(*assignment);
                if (synchronous && !localVars.hasVariable(assignment->getVariable())) {
                    continue;
                }
                if (assignment->getLevel() == 0) {
                    continue;
                }
                if (!isReadBeforeAssignment(assignment->getVariable(), upperBound(assignment->getLevel() - 1))) {
                    if (!isWrittenBeforeAssignment(assignment->getVariable(), upperBound(assignment->getLevel()-1))) {
                        newAssignments.back().setLevel(0);
                        changed = true;
                    }
                }
            }
            if (changed) {
                return OrderedAssignments(newAssignments).simplifyLevels(synchronous, localVars);
            } else {
                return *this;
            }

        }

        detail::ConstAssignments OrderedAssignments::getAllAssignments() const {
            return detail::ConstAssignments(allAssignments.begin(), allAssignments.end());
        }
        
        detail::ConstAssignments OrderedAssignments::getTransientAssignments() const {
            return detail::ConstAssignments(transientAssignments.begin(), transientAssignments.end());
        }
        
        detail::ConstAssignments OrderedAssignments::getNonTransientAssignments() const {
            return detail::ConstAssignments(nonTransientAssignments.begin(), nonTransientAssignments.end());
        }
        
        bool OrderedAssignments::hasTransientAssignment() const {
            return !transientAssignments.empty();
        }
        
        detail::Assignments::iterator OrderedAssignments::begin() {
            return detail::Assignments::make_iterator(allAssignments.begin());
        }
        
        detail::ConstAssignments::iterator OrderedAssignments::begin() const {
            return detail::ConstAssignments::make_iterator(allAssignments.begin());
        }
        
        detail::Assignments::iterator OrderedAssignments::end() {
            return detail::Assignments::make_iterator(allAssignments.end());
        }
        
        detail::ConstAssignments::iterator OrderedAssignments::end() const {
            return detail::ConstAssignments::make_iterator(allAssignments.end());
        }
        
        void OrderedAssignments::substitute(std::map<storm::expressions::Variable, storm::expressions::Expression> const& substitution) {
            for (auto& assignment : allAssignments) {
                assignment->substitute(substitution);
            }
        }
        
        void OrderedAssignments::changeAssignmentVariables(std::map<Variable const*, std::reference_wrapper<Variable const>> const& remapping) {
            std::vector<Assignment> newAssignments;
            for (auto& assignment : allAssignments) {
                newAssignments.emplace_back(remapping.at(&assignment->getVariable()), assignment->getAssignedExpression(), assignment->getLevel());
            }
            *this = OrderedAssignments(newAssignments);
        }
        
        std::vector<std::shared_ptr<Assignment>>::const_iterator OrderedAssignments::lowerBound(Assignment const& assignment, std::vector<std::shared_ptr<Assignment>> const& assignments) {
            return std::lower_bound(assignments.begin(), assignments.end(), assignment, storm::jani::AssignmentPartialOrderByLevelAndVariable());
        }

        bool OrderedAssignments::isReadBeforeAssignment(Variable const& var, uint64_t assignmentNumber, uint64_t start) const {
            for (uint64_t i = start; i < assignmentNumber; i++) {
                if (allAssignments.at(i)->getAssignedExpression().containsVariable({ var.getExpressionVariable() })) {
                    return true;
                }
            }
            return false;
        }

        bool OrderedAssignments::isWrittenBeforeAssignment(Variable const& var, uint64_t assignmentNumber, uint64_t start) const {
            for (uint64_t i = start; i < assignmentNumber; i++) {
                if (allAssignments.at(i)->getVariable() == var) {
                    return true;
                }
            }
            return false;
        }

        uint64_t OrderedAssignments::upperBound(int64_t index) const {
            uint64_t  result = 0;
            for(auto const& assignment : allAssignments) {
                if(assignment->getLevel() > index) {
                    return result;
                }
                ++result;
            }
            return result;
        }

        bool OrderedAssignments::areLinear() const {
            bool result = true;
            for (auto const& assignment : getAllAssignments()) {
                result &= assignment.isLinear();
            }
            return result;
        }
        
        std::ostream& operator<<(std::ostream& stream, OrderedAssignments const& assignments) {
            stream << "[";
            for(auto const& e : assignments.allAssignments) {
                stream << *e;
                if (e->getLevel() != 0) {
                    stream << " @" << e->getLevel();
                }
                stream << std::endl;
            }
            stream << "]";
            return stream;
        }
        
    }
}
