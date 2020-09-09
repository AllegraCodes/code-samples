package project5;

import java.util.ArrayList;

/**
 * Singleton class for input list
 * Stores all items created from file processing
 * into a global object that can be
 * accessed by all classes.
 */
public class InputList {
    // static variable single_instance of type Singleton
    private static InputList single_instance = null;

    public ArrayList<Item> itemList = new ArrayList<>();
    public ArrayList<Bag> bagList = new ArrayList<>();
    public ArrayList<Integer> fittingLimit = new ArrayList<>();
    public ArrayList<UnaryConstraint> unary = new ArrayList<>();
    public ArrayList<BinaryEqualityConstraint> binaryEquals = new ArrayList<>();
    public ArrayList<BinaryMutualConstraint> mutualInclusive = new ArrayList<>();

    // private constructor restricted to this class itself
    private InputList(ArrayList<Item> itemList,
                      ArrayList<Bag> bagList,
                      ArrayList<Integer> fittingLimit,
                      ArrayList<UnaryConstraint> unary,
                      ArrayList<BinaryEqualityConstraint> binaryEquals,
                      ArrayList<BinaryMutualConstraint> mutualInclusive)
    {
        itemList = itemList;
        bagList = bagList;
        fittingLimit = fittingLimit;
        unary = unary;
        binaryEquals = binaryEquals;
        mutualInclusive = mutualInclusive;
    }

    // static method to create instance of InputList class
    public static InputList getInputList()
    {
        if (single_instance == null) {
            single_instance =
                    new InputList(
                    new ArrayList<Item>(),
                    new ArrayList<Bag>(),
                    new ArrayList<Integer>(),
                    new ArrayList<UnaryConstraint>(),
                    new ArrayList<BinaryEqualityConstraint>(),
                    new ArrayList<BinaryMutualConstraint>());
        }
        return single_instance;
    }
}
