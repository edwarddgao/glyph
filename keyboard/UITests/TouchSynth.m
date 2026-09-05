// Arbitrary timed touch paths through XCTest's private event synthesizer —
// the mechanism Appium's WebDriverAgent uses. Public XCUITest only offers
// two-point drags, which cannot replay a swipe-typing gesture.
//
// Usage from Swift:  TouchSynth.replay(points, durationScale: 1.0, error: &err)
// where points are {x, y, t} in screen points / seconds (t relative to 0).
#import <Foundation/Foundation.h>
#import <XCTest/XCTest.h>
#import <CoreGraphics/CoreGraphics.h>
#import "TouchSynth.h"

@interface XCPointerEventPath : NSObject
- (instancetype)initForTouchAtPoint:(CGPoint)point offset:(NSTimeInterval)offset;
- (void)moveToPoint:(CGPoint)point atOffset:(NSTimeInterval)offset;
- (void)liftUpAtOffset:(NSTimeInterval)offset;
@end

@interface XCSynthesizedEventRecord : NSObject
- (instancetype)initWithName:(NSString *)name interfaceOrientation:(long long)orientation;
- (void)addPointerEventPath:(XCPointerEventPath *)path;
- (BOOL)synthesizeWithError:(NSError **)error;
@end

@interface XCUIDevice (Private)
- (id)eventSynthesizer;
@end

@interface NSObject (XCTRunnerDaemonSessionPrivate)
+ (id)sharedSession;
- (void)synthesizeEvent:(id)event completion:(void (^)(NSError *))completion;
@end

#import <objc/runtime.h>

static NSString *methodsOf(Class c) {
    unsigned n = 0; Method *ms = class_copyMethodList(c, &n);
    NSMutableArray *out = [NSMutableArray array];
    for (unsigned i = 0; i < n; i++) [out addObject:NSStringFromSelector(method_getName(ms[i]))];
    free(ms);
    return [out componentsJoinedByString:@" "];
}

@implementation TouchSynth

+ (NSString *)describePrivateAPI {
    NSMutableString *s = [NSMutableString string];
    for (NSString *name in @[@"XCPointerEventPath", @"XCSynthesizedEventRecord", @"XCTRunnerDaemonSession", @"XCUIEventSynthesizer"]) {
        Class c = NSClassFromString(name);
        [s appendFormat:@"\n%@: %@\n  instance: %@\n  class: %@\n", name, c ? @"present" : @"MISSING",
         c ? methodsOf(c) : @"", c ? methodsOf(object_getClass(c)) : @""];
    }
    return s;
}

+ (BOOL)replayPoints:(NSArray<NSValue *> *)points times:(NSArray<NSNumber *> *)times error:(NSError **)error {
    if (points.count == 0 || points.count != times.count) {
        if (error) *error = [NSError errorWithDomain:@"TouchSynth" code:1 userInfo:@{NSLocalizedDescriptionKey: @"empty or mismatched path"}];
        return NO;
    }
    Class pathCls = NSClassFromString(@"XCPointerEventPath");
    Class recCls = NSClassFromString(@"XCSynthesizedEventRecord");
    Class sessCls = NSClassFromString(@"XCTRunnerDaemonSession");
    if (!pathCls || !recCls || !sessCls) {
        if (error) *error = [NSError errorWithDomain:@"TouchSynth" code:2 userInfo:@{NSLocalizedDescriptionKey: @"private XCTest synthesizer classes unavailable"}];
        return NO;
    }
    CGPoint p0 = [points[0] CGPointValue];
    XCPointerEventPath *path = [[pathCls alloc] initForTouchAtPoint:p0 offset:0];
    NSTimeInterval last = 0;
    for (NSUInteger i = 1; i < points.count; i++) {
        NSTimeInterval t = MAX(times[i].doubleValue, last + 0.001);  // strictly increasing
        [path moveToPoint:[points[i] CGPointValue] atOffset:t];
        last = t;
    }
    [path liftUpAtOffset:last + 0.02];

    XCSynthesizedEventRecord *rec = [[recCls alloc] initWithName:@"swipe replay" interfaceOrientation:1 /* portrait */];
    [rec addPointerEventPath:path];
    // Synchronous: returns after the whole path has been delivered.
    NSError *inner = nil;
    BOOL ok = [rec synthesizeWithError:&inner];
    if (!ok) { if (error) *error = inner ?: [NSError errorWithDomain:@"TouchSynth" code:3 userInfo:@{NSLocalizedDescriptionKey: @"synthesize failed"}]; return NO; }
    return YES;
}

@end
