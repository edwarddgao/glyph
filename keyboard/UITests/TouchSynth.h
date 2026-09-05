#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Replays one finger-down → moves → lift path with explicit timing, in
/// screen points, through XCTest's private event synthesizer.
@interface TouchSynth : NSObject
+ (NSString *)describePrivateAPI;
+ (BOOL)replayPoints:(NSArray<NSValue *> *)points times:(NSArray<NSNumber *> *)times error:(NSError **)error;
@end

NS_ASSUME_NONNULL_END
