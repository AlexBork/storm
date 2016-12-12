#pragma once

#include "storm/storage/jani/OrderedAssignments.h"

namespace storm {
    namespace jani {
        
        class TemplateEdgeDestination {
        public:
            TemplateEdgeDestination() = default;
            TemplateEdgeDestination(OrderedAssignments const& assignments);
            TemplateEdgeDestination(Assignment const& assignment);
            TemplateEdgeDestination(std::vector<Assignment> const& assignments);

            /*!
             * Substitutes all variables in all expressions according to the given substitution.
             */
            void substitute(std::map<storm::expressions::Variable, storm::expressions::Expression> const& substitution);
            
            /*!
             * Changes all variables in assignments based on the given mapping.
             */
            void changeAssignmentVariables(std::map<Variable const*, std::reference_wrapper<Variable const>> const& remapping);

            OrderedAssignments const& getOrderedAssignments() const;
            
            // Convenience methods to access the assignments.
            bool hasAssignment(Assignment const& assignment) const;
            bool removeAssignment(Assignment const& assignment);
            void addAssignment(Assignment const& assignment);
            
            /*!
             * Retrieves whether this destination has transient assignments.
             */
            bool hasTransientAssignment() const;
            
            /*!
             * Retrieves whether the edge uses an assignment level other than zero.
             */
            bool usesAssignmentLevels() const;
            
        private:
            // The (ordered) assignments to make when choosing this destination.
            OrderedAssignments assignments;
        };
        
    }
}
