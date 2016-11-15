#ifndef STORM_UTILITY_VECTOR_H_
#define STORM_UTILITY_VECTOR_H_

#include "storm-config.h"
#ifdef STORM_HAVE_INTELTBB
#include "tbb/tbb.h"
#endif

#include "constants.h"
#include <iostream>
#include <algorithm>
#include <functional>

#include "src/storage/BitVector.h"
#include "src/utility/macros.h"
#include "src/solver/OptimizationDirection.h"

namespace storm {
    namespace utility {
        namespace vector {

            /*!
             * Finds the given element in the given vector.
             * If the vector does not contain the element, it is inserted (at the end of vector).
             * Either way, the returned value will be the position of the element inside the vector,
             * 
             * @note old indices to other elements remain valid, as the vector will not be sorted.
             * 
             * @param vector The vector in which the element is searched and possibly insert
             * @param element The element that will be searched for (or inserted)
             * 
             * @return The position at which the element is located
             */
            template<class T>
            std::size_t findOrInsert(std::vector<T>& vector, T&& element){
                std::size_t position=std::find(vector.begin(), vector.end(), element) - vector.begin();
                if(position==vector.size()){
                    vector.emplace_back(std::move(element));
                }
                return position;
            }
            
            /*!
             * Sets the provided values at the provided positions in the given vector.
             *
             * @param vector The vector in which the values are to be set.
             * @param positions The positions at which the values are to be set.
             * @param values The values that are to be set.
             */
            template<class T>
            void setVectorValues(std::vector<T>& vector, storm::storage::BitVector const& positions, std::vector<T> const& values) {
                uint_fast64_t oldPosition = 0;
                for (auto position : positions) {
                    vector[position] = values[oldPosition++];
                }
            }
            
            /*!
             * Sets the provided value at the provided positions in the given vector.
             *
             * @param vector The vector in which the value is to be set.
             * @param positions The positions at which the value is to be set.
             * @param value The value that is to be set.
             */
            template<class T>
            void setVectorValues(std::vector<T>& vector, storm::storage::BitVector const& positions, T value) {
                for (auto position : positions) {
                    vector[position] = value;
                }
            }

            /*!
             * Iota function as a helper for efficient creating a range in a vector.
             * See also http://stackoverflow.com/questions/11965732/set-stdvectorint-to-a-range
             * @see buildVectorForRange
             */
            template<class OutputIterator, class Size, class Assignable>
            void iota_n(OutputIterator first, Size n, Assignable value)
            {
                std::generate_n(first, n, [&value]() {
                    return value++;
                });
            }

            /*!
             * Constructs a vector [min, min+1, ...., max]
             */
            inline std::vector<uint_fast64_t> buildVectorForRange(uint_fast64_t min, uint_fast64_t max) {
                STORM_LOG_ASSERT(min < max, "Invalid range.");
                uint_fast64_t diff = max - min;
                std::vector<uint_fast64_t> v;
                v.reserve(diff);
                iota_n(std::back_inserter(v), diff, min);
                return v;
            }


                        
            /*!
             * Selects the elements from a vector at the specified positions and writes them consecutively into another vector.
             * @param vector The vector into which the selected elements are to be written.
             * @param positions The positions at which to select the elements from the values vector.
             * @param values The vector from which to select the elements.
             */
            template<class T>
            void selectVectorValues(std::vector<T>& vector, storm::storage::BitVector const& positions, std::vector<T> const& values) {
                uint_fast64_t oldPosition = 0;
                for (auto position : positions) {
                    vector[oldPosition++] = values[position];
                }
            }
            
            /*!
             * Selects groups of elements from a vector at the specified positions and writes them consecutively into another vector.
             *
             * @param vector The vector into which the selected elements are to be written.
             * @param positions The positions of the groups of elements that are to be selected.
             * @param rowGrouping A vector that specifies the begin and end of each group of elements in the values vector.
             * @param values The vector from which to select groups of elements.
             */
            template<class T>
            void selectVectorValues(std::vector<T>& vector, storm::storage::BitVector const& positions, std::vector<uint_fast64_t> const& rowGrouping, std::vector<T> const& values) {
                uint_fast64_t oldPosition = 0;
                for (auto position : positions) {
                    for (uint_fast64_t i = rowGrouping[position]; i < rowGrouping[position + 1]; ++i) {
                        vector[oldPosition++] = values[i];
                    }
                }
            }
            
            /*!
             * Selects one element out of each row group and writes it to the target vector.
             *
             * @param vector The target vector to which the values are written.
             * @param rowGroupToRowIndexMapping A mapping from row group indices to an offset that specifies which of the values to
             * take from the row group.
             * @param rowGrouping A vector that specifies the begin and end of each group of elements in the values vector.
             * @param values The vector from which to select the values.
             */
            template<class T>
            void selectVectorValues(std::vector<T>& vector, std::vector<uint_fast64_t> const& rowGroupToRowIndexMapping, std::vector<uint_fast64_t> const& rowGrouping, std::vector<T> const& values) {
                for (uint_fast64_t i = 0; i < vector.size(); ++i) {
                    vector[i] = values[rowGrouping[i] + rowGroupToRowIndexMapping[i]];
                }
            }
            
            /*!
             * Selects values from a vector at the specified sequence of indices and writes them into another vector
             *
             * @param vector The vector into which the selected elements are written.
             * @param indexSequence a sequence of indices at which the desired values can be found
             * @param values the values from which to select
             */
            template<class T>
            void selectVectorValues(std::vector<T>& vector, std::vector<uint_fast64_t> const& indexSequence, std::vector<T> const& values) {
                for (uint_fast64_t vectorIndex = 0; vectorIndex < vector.size(); ++vectorIndex){
                    vector[vectorIndex] = values[indexSequence[vectorIndex]];
                }
            }
            
            /*!
             * Selects values from a vector at the specified positions and writes them into another vector as often as given by
             * the size of the corresponding group of elements.
             *
             * @param vector The vector into which the selected elements are written.
             * @param positions The positions at which to select the values.
             * @param rowGrouping A vector that specifies the begin and end of each group of elements in the values vector. This
             * implicitly defines the number of times any element is written to the output vector.
             */
            template<class T>
            void selectVectorValuesRepeatedly(std::vector<T>& vector, storm::storage::BitVector const& positions, std::vector<uint_fast64_t> const& rowGrouping, std::vector<T> const& values) {
                uint_fast64_t oldPosition = 0;
                for (auto position : positions) {
                    for (uint_fast64_t i = rowGrouping[position]; i < rowGrouping[position + 1]; ++i) {
                        vector[oldPosition++] = values[position];
                    }
                }
            }
            
            /*!
             * Subtracts the given vector from the constant one-vector and writes the result to the input vector.
             *
             * @param vector The vector that is to be subtracted from the constant one-vector.
             */
            template<class T>
            void subtractFromConstantOneVector(std::vector<T>& vector) {
                for (auto& element : vector) {
                    element = storm::utility::one<T>() - element;
                }
            }
            
            template<class T>
            void addFilteredVectorGroupsToGroupedVector(std::vector<T>& target, std::vector<T> const& source, storm::storage::BitVector const& filter, std::vector<uint_fast64_t> const& rowGroupIndices) {
                uint_fast64_t currentPosition = 0;
                for (auto group : filter) {
                    auto it = source.cbegin() + rowGroupIndices[group];
                    auto ite = source.cbegin() + rowGroupIndices[group + 1];
                    while (it != ite) {
                        target[currentPosition] += *it;
                        ++currentPosition;
                        ++it;
                    }
                }
            }
            
            /*!
             * Adds the source vector to the target vector in a way such that the i-th entry is added to all elements of
             * the i-th row group in the target vector.
             *
             * @param target The target ("row grouped") vector.
             * @param source The source vector.
             * @param rowGroupIndices A vector representing the row groups in the target vector.
             */
            template<class T>
            void addVectorToGroupedVector(std::vector<T>& target, std::vector<T> const& source, std::vector<uint_fast64_t> const& rowGroupIndices) {
                auto targetIt = target.begin();
                auto sourceIt = source.cbegin();
                auto sourceIte = source.cend();
                auto rowGroupIt = rowGroupIndices.cbegin();
                
                for (; sourceIt != sourceIte; ++sourceIt) {
                    uint_fast64_t current = *rowGroupIt;
                    ++rowGroupIt;
                    uint_fast64_t next = *rowGroupIt;
                    while (current < next) {
                        *targetIt += *sourceIt;
                        ++targetIt;
                        ++current;
                    }
                }
            }
            
            /*!
             * Adds the source vector to the target vector in a way such that the i-th selected entry is added to all
             * elements of the i-th row group in the target vector.
             *
             * @param target The target ("row grouped") vector.
             * @param source The source vector.
             * @param rowGroupIndices A vector representing the row groups in the target vector.
             */
            template<class T>
            void addFilteredVectorToGroupedVector(std::vector<T>& target, std::vector<T> const& source, storm::storage::BitVector const& filter, std::vector<uint_fast64_t> const& rowGroupIndices) {
                uint_fast64_t currentPosition = 0;
                for (auto group : filter) {
                    uint_fast64_t current = rowGroupIndices[group];
                    uint_fast64_t next = rowGroupIndices[group + 1];
                    while (current < next) {
                        target[currentPosition] += source[group];
                        ++currentPosition;
                        ++current;
                    }
                }
            }
            
            /*!
             * Applies the given operation pointwise on the two given vectors and writes the result to the third vector.
             * To obtain an in-place operation, the third vector may be equal to any of the other two vectors.
             *
             * @param firstOperand The first operand.
             * @param secondOperand The second operand.
             * @param target The target vector.
             */
            template<class InValueType1, class InValueType2, class OutValueType>
            void applyPointwise(std::vector<InValueType1> const& firstOperand, std::vector<InValueType2> const& secondOperand, std::vector<OutValueType>& target, std::function<OutValueType (InValueType1 const&, InValueType2 const&, OutValueType const&)> const& function) {
#ifdef STORM_HAVE_INTELTBB
                tbb::parallel_for(tbb::blocked_range<uint_fast64_t>(0, target.size()),
                                  [&](tbb::blocked_range<uint_fast64_t> const& range) {
                                      auto firstIt = firstOperand.begin() + range.begin();
                                      auto firstIte = firstOperand.begin() + range.end();
                                      auto secondIt = secondOperand.begin() + range.begin();
                                      auto targetIt = target.begin() + range.begin();
                                      while (firstIt != firstIte) {
                                          *targetIt = function(*firstIt, *secondIt, *targetIt);
                                          ++targetIt;
                                          ++firstIt;
                                          ++secondIt;
                                      }
                                  });
#else
                auto firstIt = firstOperand.begin();
                auto firstIte = firstOperand.end();
                auto secondIt = secondOperand.begin();
                auto targetIt = target.begin();
                while (firstIt != firstIte) {
                    *targetIt = function(*firstIt, *secondIt, *targetIt);
                    ++targetIt;
                    ++firstIt;
                    ++secondIt;
                }
#endif
            }
            
            /*!
             * Applies the given operation pointwise on the two given vectors and writes the result to the third vector.
             * To obtain an in-place operation, the third vector may be equal to any of the other two vectors.
             *
             * @param firstOperand The first operand.
             * @param secondOperand The second operand.
             * @param target The target vector.
             */
            template<class InValueType1, class InValueType2, class OutValueType>
            void applyPointwise(std::vector<InValueType1> const& firstOperand, std::vector<InValueType2> const& secondOperand, std::vector<OutValueType>& target, std::function<OutValueType (InValueType1 const&, InValueType2 const&)> const& function) {
#ifdef STORM_HAVE_INTELTBB
                tbb::parallel_for(tbb::blocked_range<uint_fast64_t>(0, target.size()),
                                  [&](tbb::blocked_range<uint_fast64_t> const& range) {
                                      std::transform(firstOperand.begin() + range.begin(), firstOperand.begin() + range.end(), secondOperand.begin() + range.begin(), target.begin() + range.begin(), function);
                                  });
#else
                std::transform(firstOperand.begin(), firstOperand.end(), secondOperand.begin(), target.begin(), function);
#endif
            }
            
            /*!
             * Applies the given function pointwise on the given vector.
             *
             * @param operand The vector to which to apply the function.
             * @param target The target vector.
             * @param function The function to apply.
             */
            template<class InValueType, class OutValueType>
            void applyPointwise(std::vector<InValueType> const& operand, std::vector<OutValueType>& target, std::function<OutValueType (InValueType const&)> const& function) {
#ifdef STORM_HAVE_INTELTBB
                tbb::parallel_for(tbb::blocked_range<uint_fast64_t>(0, target.size()),
                                  [&](tbb::blocked_range<uint_fast64_t> const& range) {
                                      std::transform(operand.begin() + range.begin(), operand.begin() + range.end(), target.begin() + range.begin(), function);
                                  });
#else
                std::transform(operand.begin(), operand.end(), target.begin(), function);
#endif
            }
            
            /*!
             * Adds the two given vectors and writes the result to the target vector.
             *
             * @param firstOperand The first operand.
             * @param secondOperand The second operand
             * @param target The target vector.
             */
            template<class InValueType1, class InValueType2, class OutValueType>
            void addVectors(std::vector<InValueType1> const& firstOperand, std::vector<InValueType2> const& secondOperand, std::vector<OutValueType>& target) {
                applyPointwise<InValueType1, InValueType2, OutValueType>(firstOperand, secondOperand, target, std::plus<>());
            }
            
            /*!
             * Subtracts the two given vectors and writes the result to the target vector.
             *
             * @param firstOperand The first operand.
             * @param secondOperand The second operand
             * @param target The target vector.
             */
            template<class InValueType1, class InValueType2, class OutValueType>
            void subtractVectors(std::vector<InValueType1> const& firstOperand, std::vector<InValueType2> const& secondOperand, std::vector<OutValueType>& target) {
                applyPointwise<InValueType1, InValueType2, OutValueType>(firstOperand, secondOperand, target, std::minus<>());
            }
            
            /*!
             * Multiplies the two given vectors (pointwise) and writes the result to the target vector.
             *
             * @param firstOperand The first operand.
             * @param secondOperand The second operand
             * @param target The target vector.
             */
            template<class InValueType1, class InValueType2, class OutValueType>
            void multiplyVectorsPointwise(std::vector<InValueType1> const& firstOperand, std::vector<InValueType2> const& secondOperand, std::vector<OutValueType>& target) {
                applyPointwise<InValueType1, InValueType2, OutValueType>(firstOperand, secondOperand, target, std::multiplies<>());
            }
            
            /*!
             * Subtracts the two given vectors and writes the result into the first operand.
             *
             * @param target The first summand and target vector.
             * @param summand The second summand.
             */
            template<class ValueType1, class ValueType2>
            void scaleVectorInPlace(std::vector<ValueType1>& target, ValueType2 const& factor) {
                applyPointwise<ValueType1, ValueType2>(target, target, [&] (ValueType1 const& argument) -> ValueType1 { return argument * factor; });
            }
            
            /*!
             * Retrieves a bit vector containing all the indices for which the value at this position makes the given
             * function evaluate to true.
             *
             * @param values The vector of values.
             * @param function The function that selects some elements.
             * @return The resulting bit vector.
             */
            template<class T>
            storm::storage::BitVector filter(std::vector<T> const& values, std::function<bool (T const& value)> const& function) {
                storm::storage::BitVector result(values.size());
                
                uint_fast64_t currentIndex = 0;
                for (auto const& value : values) {
                    result.set(currentIndex, function(value));
                    ++currentIndex;
                }
                
                return result;
            }
            
            /*!
             * Retrieves a bit vector containing all the indices for which the value at this position is greater than
             * zero
             *
             * @param values The vector of values.
             * @return The resulting bit vector.
             */
            template<class T>
            storm::storage::BitVector filterGreaterZero(std::vector<T> const& values) {
                std::function<bool (T const&)> fnc = [] (T const& value) -> bool { return value > storm::utility::zero<T>(); };
                return filter(values,  fnc);
            }
            
            /**
             * Sum the entries from values that are set to one in the filter vector.
             * @param values
             * @param filter
             * @return The sum of the values with a corresponding one in the filter.
             */
            template<typename VT>
            VT sum_if(std::vector<VT> const& values, storm::storage::BitVector const& filter) {
                STORM_LOG_ASSERT(values.size() == filter.size(), "Vector sizes mismatch.");
                VT sum = storm::utility::zero<VT>();
                for(auto pos : filter) {
                    sum += values[pos];
                }    
                return sum;
            }
            
            /**
             * Computes the maximum of the entries from the values that are selected by the (non-empty) filter.
             * @param values The values in which to search.
             * @param filter The filter to use.
             * @return The maximum over the selected values.
             */
            template<typename VT>
            VT max_if(std::vector<VT> const& values, storm::storage::BitVector const& filter) {
                STORM_LOG_ASSERT(values.size() == filter.size(), "Vector sizes mismatch.");
                STORM_LOG_ASSERT(!filter.empty(), "Empty selection.");

                auto it = filter.begin();
                auto ite = filter.end();

                VT current = values[*it];
                ++it;
                
                for (; it != ite; ++it) {
                    if (values[*it] > current) {
                        current = values[*it];
                    }
                }
                return current;
            }
            
            /**
             * Computes the minimum of the entries from the values that are selected by the (non-empty) filter.
             * @param values The values in which to search.
             * @param filter The filter to use.
             * @return The minimum over the selected values.
             */
            template<typename VT>
            VT min_if(std::vector<VT> const& values, storm::storage::BitVector const& filter) {
                STORM_LOG_ASSERT(values.size() == filter.size(), "Vector sizes mismatch.");
                STORM_LOG_ASSERT(!filter.empty(), "Empty selection.");
                
                auto it = filter.begin();
                auto ite = filter.end();
                
                VT current = values[*it];
                ++it;
                
                for (; it != ite; ++it) {
                    if (values[*it] < current) {
                        current = values[*it];
                    }
                }
                return current;
            }
            
            /*!
             * Reduces the given source vector by selecting an element according to the given filter out of each row group.
             *
             * @param source The source vector which is to be reduced.
             * @param target The target vector into which a single element from each row group is written.
             * @param rowGrouping A vector that specifies the begin and end of each group of elements in the values vector.
             * @param filter A function that compares two elements v1 and v2 according to some filter criterion. This function must
             * return true iff v1 is supposed to be taken instead of v2.
             * @param choices If non-null, this vector is used to store the choices made during the selection.
             */
            template<class T>
            void reduceVector(std::vector<T> const& source, std::vector<T>& target, std::vector<uint_fast64_t> const& rowGrouping, std::function<bool (T const&, T const&)> filter, std::vector<uint_fast64_t>* choices) {
#ifdef STORM_HAVE_INTELTBB
                tbb::parallel_for(tbb::blocked_range<uint_fast64_t>(0, target.size()),
                                  [&](tbb::blocked_range<uint_fast64_t> const& range) {
                                      uint_fast64_t startRow = range.begin();
                                      uint_fast64_t endRow = range.end();
                                      
                                      typename std::vector<T>::iterator targetIt = target.begin() + startRow;
                                      typename std::vector<T>::iterator targetIte = target.begin() + endRow;
                                      typename std::vector<uint_fast64_t>::const_iterator rowGroupingIt = rowGrouping.begin() + startRow;
                                      typename std::vector<T>::const_iterator sourceIt = source.begin() + *rowGroupingIt;
                                      typename std::vector<T>::const_iterator sourceIte;
                                      typename std::vector<uint_fast64_t>::iterator choiceIt;
                                      uint_fast64_t localChoice;
                                      if (choices != nullptr) {
                                          choiceIt = choices->begin();
                                      }

                                      for (; targetIt != targetIte; ++targetIt, ++rowGroupingIt) {
                                          *targetIt = *sourceIt;
                                          ++sourceIt;
                                          localChoice = 1;
                                          if (choices != nullptr) {
                                              *choiceIt = 0;
                                          }
                                          
                                          for (sourceIte = source.begin() + *(rowGroupingIt + 1); sourceIt != sourceIte; ++sourceIt, ++localChoice) {
                                              if (filter(*sourceIt, *targetIt)) {
                                                  *targetIt = *sourceIt;
                                                  if (choices != nullptr) {
                                                      *choiceIt = localChoice;
                                                  }
                                              }
                                          }
                                          
                                          if (choices != nullptr) {
                                              ++choiceIt;
                                          }
                                      }
                                  });
#else
                typename std::vector<T>::iterator targetIt = target.begin();
                typename std::vector<T>::iterator targetIte = target.end();
                typename std::vector<uint_fast64_t>::const_iterator rowGroupingIt = rowGrouping.begin();
                typename std::vector<T>::const_iterator sourceIt = source.begin();
                typename std::vector<T>::const_iterator sourceIte;
                typename std::vector<uint_fast64_t>::iterator choiceIt;
                uint_fast64_t localChoice;
                if (choices != nullptr) {
                    choiceIt = choices->begin();
                }
                
                for (; targetIt != targetIte; ++targetIt, ++rowGroupingIt) {
                    *targetIt = *sourceIt;
                    ++sourceIt;
                    localChoice = 1;
                    if (choices != nullptr) {
                        *choiceIt = 0;
                    }
                    for (sourceIte = source.begin() + *(rowGroupingIt + 1); sourceIt != sourceIte; ++sourceIt, ++localChoice) {
                        if (filter(*sourceIt, *targetIt)) {
                            *targetIt = *sourceIt;
                            if (choices != nullptr) {
                                *choiceIt = localChoice;
                            }
                        }
                    }
                    if (choices != nullptr) {
                        ++choiceIt;
                    }
                }
#endif
            }
                        
            /*!
             * Reduces the given source vector by selecting the smallest element out of each row group.
             *
             * @param source The source vector which is to be reduced.
             * @param target The target vector into which a single element from each row group is written.
             * @param rowGrouping A vector that specifies the begin and end of each group of elements in the source vector.
             * @param choices If non-null, this vector is used to store the choices made during the selection.
             */
            template<class T>
            void reduceVectorMin(std::vector<T> const& source, std::vector<T>& target, std::vector<uint_fast64_t> const& rowGrouping, std::vector<uint_fast64_t>* choices = nullptr) {
                reduceVector<T>(source, target, rowGrouping, std::less<T>(), choices);
            }
            
            /*!
             * Reduces the given source vector by selecting the largest element out of each row group.
             *
             * @param source The source vector which is to be reduced.
             * @param target The target vector into which a single element from each row group is written.
             * @param rowGrouping A vector that specifies the begin and end of each group of elements in the source vector.
             * @param choices If non-null, this vector is used to store the choices made during the selection.
             */
            template<class T>
            void reduceVectorMax(std::vector<T> const& source, std::vector<T>& target, std::vector<uint_fast64_t> const& rowGrouping, std::vector<uint_fast64_t>* choices = nullptr) {
                reduceVector<T>(source, target, rowGrouping, std::greater<T>(), choices);
            }
            
            /*!
             * Reduces the given source vector by selecting either the smallest or the largest out of each row group.
             *
             * @param dir If true, select the smallest, else select the largest.
             * @param source The source vector which is to be reduced.
             * @param target The target vector into which a single element from each row group is written.
             * @param rowGrouping A vector that specifies the begin and end of each group of elements in the source vector.
             * @param choices If non-null, this vector is used to store the choices made during the selection.
             */
            template<class T>
            void reduceVectorMinOrMax(storm::solver::OptimizationDirection dir, std::vector<T> const& source, std::vector<T>& target, std::vector<uint_fast64_t> const& rowGrouping, std::vector<uint_fast64_t>* choices = nullptr) {
                if(dir == storm::solver::OptimizationDirection::Minimize) {
                    reduceVectorMin(source, target, rowGrouping, choices);
                } else {
                    reduceVectorMax(source, target, rowGrouping, choices);    
                }
            }
            
            
            /*!
             * Compares the given elements and determines whether they are equal modulo the given precision. The provided flag
             * additionaly specifies whether the error is computed in relative or absolute terms.
             *
             * @param val1 The first value to compare.
             * @param val2 The second value to compare.
             * @param precision The precision up to which the elements are compared.
             * @param relativeError If set, the error is computed relative to the second value.
             * @return True iff the elements are considered equal.
             */
            template<class T>
            bool equalModuloPrecision(T const& val1, T const& val2, T const& precision, bool relativeError = true) {
                if (relativeError) {
					if (val2 == 0) {
                        return (storm::utility::abs(val1) <= precision);
					}
                    T relDiff = (val1 - val2)/val2;
                    if (storm::utility::abs(relDiff) > precision) {
                        return false;
                    }
                } else {
                    T diff = val1 - val2;
                    if (storm::utility::abs(diff) > precision) return false;
                }
                return true;
            }
            
            /*!
             * Compares the two vectors and determines whether they are equal modulo the provided precision. Depending on whether the
             * flag is set, the difference between the vectors is computed relative to the value or in absolute terms.
             *
             * @param vectorLeft The first vector of the comparison.
             * @param vectorRight The second vector of the comparison.
             * @param precision The precision up to which the vectors are to be checked for equality.
             * @param relativeError If set, the difference between the vectors is computed relative to the value or in absolute terms.
             */
            template<class T>
            bool equalModuloPrecision(std::vector<T> const& vectorLeft, std::vector<T> const& vectorRight, T const& precision, bool relativeError) {
                STORM_LOG_ASSERT(vectorLeft.size() == vectorRight.size(), "Lengths of vectors does not match.");
                
                for (uint_fast64_t i = 0; i < vectorLeft.size(); ++i) {
                    if (!equalModuloPrecision(vectorLeft[i], vectorRight[i], precision, relativeError)) {
                        return false;
                    }
                }
                
                return true;
            }
            
            /*!
             * Compares the two vectors at the specified positions and determines whether they are equal modulo the provided
             * precision. Depending on whether the flag is set, the difference between the vectors is computed relative to the value
             * or in absolute terms.
             *
             * @param vectorLeft The first vector of the comparison.
             * @param vectorRight The second vector of the comparison.
             * @param precision The precision up to which the vectors are to be checked for equality.
             * @param positions A vector representing a set of positions at which the vectors are compared.
             * @param relativeError If set, the difference between the vectors is computed relative to the value or in absolute terms.
             */
            template<class T>
            bool equalModuloPrecision(std::vector<T> const& vectorLeft, std::vector<T> const& vectorRight, std::vector<uint_fast64_t> const& positions, T const& precision, bool relativeError) {
                STORM_LOG_ASSERT(vectorLeft.size() == vectorRight.size(), "Lengths of vectors does not match.");
                
                for (uint_fast64_t position : positions) {
                    if (!equalModuloPrecision(vectorLeft[position], vectorRight[position], precision, relativeError)) {
                        return false;
                    }
                }
                
                return true;
            }
            
            /*!
             * Takes the given offset vector and applies the given contraint. That is, it produces another offset vector that contains
             * the relative offsets of the entries given by the constraint.
             *
             * @param offsetVector The offset vector to constrain.
             * @param constraint The constraint to apply to the offset vector.
             * @return An offset vector that contains all selected relative offsets.
             */
            template<class T>
            std::vector<T> getConstrainedOffsetVector(std::vector<T> const& offsetVector, storm::storage::BitVector const& constraint) {
                // Reserve the known amount of slots for the resulting vector.
                std::vector<uint_fast64_t> subVector(constraint.getNumberOfSetBits() + 1);
                uint_fast64_t currentRowCount = 0;
                uint_fast64_t currentIndexCount = 1;
                
                // Set the first element as this will clearly begin at offset 0.
                subVector[0] = 0;
                
                // Loop over all states that need to be kept and copy the relative indices of the nondeterministic choices over
                // to the resulting vector.
                for (auto index : constraint) {
                    subVector[currentIndexCount] = currentRowCount + offsetVector[index + 1] - offsetVector[index];
                    currentRowCount += offsetVector[index + 1] - offsetVector[index];
                    ++currentIndexCount;
                }
                
                // Put a sentinel element at the end.
                subVector[constraint.getNumberOfSetBits()] = currentRowCount;
                
                return subVector;
            }

			/*!
			* Converts the given vector to the given ValueType
			*/
			template<typename NewValueType, typename ValueType>
			std::vector<NewValueType> toValueType(std::vector<ValueType> const& oldVector) {
				std::vector<NewValueType> resultVector;
				resultVector.resize(oldVector.size());
				for (size_t i = 0, size = oldVector.size(); i < size; ++i) {
					resultVector.at(i) = static_cast<NewValueType>(oldVector.at(i));
				}

				return resultVector;
			}

            template<typename Type>
            std::vector<Type> filterVector(std::vector<Type> const& in, storm::storage::BitVector const& filter) {
                std::vector<Type> result;
                result.reserve(filter.getNumberOfSetBits());
                for(auto index : filter) {
                    result.push_back(in[index]);
                }
                STORM_LOG_ASSERT(result.size() == filter.getNumberOfSetBits(), "Result does not match.");
                return result;
            }
            
            /*!
             * Output vector as string.
             *
             * @param vector Vector to output.
             * @return String containing the representation of the vector.
             */
            template<typename ValueType>
            std::string toString(std::vector<ValueType> vector) {
                std::stringstream stream;
                stream << "vector (" << vector.size() << ") [ ";
                if (!vector.empty()) {
                    for (uint_fast64_t i = 0; i < vector.size() - 1; ++i) {
                        stream << vector[i] << ", ";
                    }
                    stream << vector.back();
                }
                stream << " ]";
                return stream.str();
            }
        } // namespace vector
    } // namespace utility
} // namespace storm

#endif /* STORM_UTILITY_VECTOR_H_ */
