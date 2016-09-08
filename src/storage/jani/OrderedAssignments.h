#pragma once

#include "src/adapters/DereferenceIteratorAdapter.h"

#include "src/storage/jani/Assignment.h"

namespace storm {
    namespace jani {
                
        namespace detail {
            using Assignments = storm::adapters::DereferenceIteratorAdapter<std::vector<std::shared_ptr<Assignment>>>;
            using ConstAssignments = storm::adapters::DereferenceIteratorAdapter<std::vector<std::shared_ptr<Assignment>> const>;
        }
        
        class OrderedAssignments {
        public:
            /*!
             * Creates an ordered set of assignments.
             */
            OrderedAssignments(std::vector<Assignment> const& assignments = std::vector<Assignment>());

            /*!
             * Adds the given assignment to the set of assignments.
             *
             * @return True iff the assignment was added.
             */
            bool add(Assignment const& assignment);
            
            /*!
             * Removes the given assignment from this set of assignments.
             *
             * @return True if the assignment was found and removed.
             */
            bool remove(Assignment const& assignment);

            /*!
             * Retrieves whether the given assignment is contained in this set of assignments.
             */
            bool contains(Assignment const& assignment) const;
            
            /*!
             * Returns all assignments in this set of assignments.
             */
            detail::ConstAssignments getAllAssignments() const;

            /*!
             * Returns all transient assignments in this set of assignments.
             */
            detail::ConstAssignments getTransientAssignments() const;

            /*!
             * Returns all non-transient assignments in this set of assignments.
             */
            detail::ConstAssignments getNonTransientAssignments() const;

            /*!
             * Returns an iterator to the assignments.
             */
            detail::Assignments::iterator begin();

            /*!
             * Returns an iterator to the assignments.
             */
            detail::ConstAssignments::iterator begin() const;

            /*!
             * Returns an iterator past the end of the assignments.
             */
            detail::Assignments::iterator end();

            /*!
             * Returns an iterator past the end of the assignments.
             */
            detail::ConstAssignments::iterator end() const;
            
            /*!
             * Substitutes all variables in all expressions according to the given substitution.
             */
            void substitute(std::map<storm::expressions::Variable, storm::expressions::Expression> const& substitution);

        private:
            static std::vector<std::shared_ptr<Assignment>>::const_iterator lowerBound(Assignment const& assignment, std::vector<std::shared_ptr<Assignment>> const& assignments);
                        
            // The vectors to store the assignments. These need to be ordered at all times.
            std::vector<std::shared_ptr<Assignment>> allAssignments;
            std::vector<std::shared_ptr<Assignment>> transientAssignments;
            std::vector<std::shared_ptr<Assignment>> nonTransientAssignments;
        };
        
    }
}