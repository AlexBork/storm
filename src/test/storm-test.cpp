#include "gtest/gtest.h"
#include "storm/settings/SettingsManager.h"

int main(int argc, char **argv) {
  storm::settings::initializeAll("storm (Functional) Testing Suite", "test");
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
