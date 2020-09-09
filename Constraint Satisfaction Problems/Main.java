import project5.*;
import java.io.*;
import java.util.ArrayList;

public class Main {
    /**
     * Compile using command: javac Main.java
     * Run using command: java -cp ./ Main [input file]
     */
    public static void main(String [] args) {
        ArrayList<Item> items = new ArrayList<>();
        ArrayList<Bag> bags = new ArrayList<>();
        BagConstraint bc = new BagConstraint();
        //InputList inputList = InputList.getInputList();

        //reading in file from command line
        //if no input, using test input1.txt file
        File file = null;
        if (0 < args.length) {
            file = new File(args[0]);
        } else {
            file = new File("input1.txt");
        }

        BufferedReader r = null;

        try {
            String line;
            r = new BufferedReader(new FileReader(file));
            ArrayList<String> lines = new ArrayList<String>();

            while ((line = r.readLine()) != null) {
                lines.add(line);
            }

            //process lines of file
            process(lines, items, bags, bc);


        } catch (IOException e) {
            e.printStackTrace();
        }

        finally {
            try {
                if (r != null) {
                    r.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        /**
         * For testing purposes, hard coding the input for input1.txt
         *
        inputList.itemList.add(new Item('a', 10));
        inputList.bagList.add(new Bag('a', 10));

        for (Item i : inputList.itemList) {
            int count = 0;
            System.out.println("Item " + count + " is: " + i.getName());
            System.out.println("Bag '" + inputList.bagList.get(0).getName() + "' has capacity " + inputList.bagList.get(0).getCapacity());
            count++;
        }*/

        Solver solver = new Solver(items, bags, bc);
        ArrayList<Bag> solution = solver.solve(false, false, false, false);
        displaySolution(solution);
    }

    private static void displaySolution(ArrayList<Bag> solution) {
        if (solution != null) {
            for (Bag b : solution) {
                printBlock(b);
            }
        } else {
            System.out.println("No solution");
        }
    }

    private static void printBlock(Bag b) {
        ArrayList<Item> contents = b.getContents();
        // line 1
        System.out.print(b.getName() + " ");
        for (Item i : contents) {
            System.out.print(i.getName() + " ");
        }
        System.out.print("\n");
        // line 2
        System.out.println("number of items: " + contents.size());
        // line 3
        System.out.println("total weight: " + b.getWeight() + "/" + b.getCapacity());
        // line 4
        System.out.println("wasted capacity: " + (b.getCapacity() - b.getWeight()));
        // blank line
        System.out.println();
    }

    /**
     * Processes the lines of the files and creates
     * the corresponding items, bags, and constraints.
     * @param lines
     */
    private static void process(ArrayList<String> lines,
                                  ArrayList<Item> items,
                                  ArrayList<Bag> bags,
                                  BagConstraint bc) {
        int section = 0;
        for (String line : lines) {
            if (line.contains("#")) {
                section++;
                continue;
            }
            String[] words = line.split(" ");
            switch (section) {
                case(1): // names and weights of the items
                    items.add(new Item(words[0], words[1]));
                    break;
                case(2): // names of the bags and their capacity
                    bags.add(new Bag(words[0], words[1]));
                    break;
                case(3): // fit limits (min, max)
                    bc.setMin(Integer.parseInt(words[0]));
                    bc.setMax(Integer.parseInt(words[1]));
                    break;
                case(4): // inclusive unary constraints
                    makeUnaryConstraint(true, words, items, bags);
                    break;
                case(5): // exclusive unary constraints
                    makeUnaryConstraint(false, words, items, bags);
                    break;
                case(6): // equal binary constraints
                    makeBinaryEqualityConstraint(true, words, items, bags);
                    break;
                case(7): // not equal binary constraints
                    makeBinaryEqualityConstraint(false, words, items, bags);
                    break;
                case(8): // mutual inclusive binary constraints
                    Item item1 = findItem(words[0], items);
                    Item item2 = findItem(words[1], items);
                    Bag bag1 = findBag(words[2], bags);
                    Bag bag2 = findBag(words[3], bags);
                    BinaryMutualConstraint bmc = new BinaryMutualConstraint(item1, item2, bag1, bag2);
                    item1.addConstraint(bmc);
                    item2.addConstraint(bmc);
                    break;
            }
        }
    }

    private static void makeBinaryEqualityConstraint(boolean equality, String[] words,
                                                     ArrayList<Item> items, ArrayList<Bag> bags) {
        ArrayList<Item> itemList = new ArrayList<>();
        for (int i = 0; i < words.length; i++) {
            itemList.add(findItem(words[i], items));
        }
        for (int i = 0; i < itemList.size(); i++) {
            if (i+1 == itemList.size()) {
                Item item1 = itemList.get(i);
                Item item2 = itemList.get(0);
                BinaryEqualityConstraint bec = new BinaryEqualityConstraint(item1, item2, equality);
                item1.addConstraint(bec);
                item2.addConstraint(bec);
            } else {
                Item item1 = itemList.get(i);
                Item item2 = itemList.get(i+1);
                BinaryEqualityConstraint bec = new BinaryEqualityConstraint(item1, item2, equality);
                item1.addConstraint(bec);
                item2.addConstraint(bec);
            }
        }
    }

    private static void makeUnaryConstraint(boolean inclusive, String[] words,
                                                       ArrayList<Item> items, ArrayList<Bag> bags) {
        Item item = findItem(words[0], items); // find the item
        // get the bag names
        ArrayList<String> bagNames = new ArrayList<>();
        for (int i = 1; i < words.length; i++) {
            bagNames.add(words[i]);
        }
        // find the bags
        ArrayList<Bag> bagList = findBags(bagNames, bags);
        // make the constraint
        UnaryConstraint uc = new UnaryConstraint(item, bagList, inclusive);
        item.addConstraint(uc);
    }

    private static Item findItem(String name, ArrayList<Item> items) {
        Item item = new Item("", 0);
        for (Item i : items) {
            if (name.equals(i.getName())) item = i;
        }
        return item;
    }

    private static Bag findBag(String name, ArrayList<Bag> bags) {
        Bag bag = new Bag("", 0);
        for (Bag b : bags) {
            if (name.equals(b.getName())) return b;
        }
        return bag;
    }

    private static ArrayList<Bag> findBags(ArrayList<String> names, ArrayList<Bag> bags) {
        ArrayList<Bag> found = new ArrayList<>();
        for (Bag b : bags) {
            if (names.contains(b.getName())) found.add(b);
        }
        return found;
    }
}