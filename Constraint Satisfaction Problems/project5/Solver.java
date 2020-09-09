package project5;

import java.util.ArrayList;

public class Solver {
    ArrayList<Item> items;
    ArrayList<Bag> bags;
    BagConstraint bc;

    public Solver(ArrayList<Item> i, ArrayList<Bag> b, BagConstraint c) {
        this.items = i;
        this.bags = b;
        this.bc = c;
        for (Item item : i) {
            for (Bag bag : b) {
                item.domain.add(bag);
            }
        }
    }

    /**
     * Solves the CSP given by items, bags, and bc using a backtracking search.
     * @return list of bags representing a full solution, or null if no solution exists
     */
    public ArrayList<Bag> solve(boolean mrv, boolean lcv, boolean deg, boolean fc) {
        if (allAssigned()) return bags; // if assignment is complete, return bags
        Item var = pickItem(mrv, deg); // choose an item using Minimum Remaining Values heuristic
        for (Bag b : orderBags(var, lcv)) { // order assignments using Least Constraining Value heuristic
            b.addItem(var);
            boolean consistent = true;
            if (!bc.check(items, bags)) { // check constraints on bags
                consistent = false;
            } else { // check constraints on item
                for (Constraint c : var.constraints) {
                    if (!c.check(items, bags)) {
                        consistent = false;
                    }
                }
            }
            if (consistent) { // keep going
                if (fc) forwardChecking(var);
                ArrayList<Bag> solution = solve(mrv, lcv, deg, fc);
                if (solution != null) {
                    return solution;
                } else {
                    b.removeItem(var);
                }
            } else { // backtrack
                b.removeItem(var);
            }
        }
        // failure
        return null;
    }

    private Item pickItem(boolean mrv, boolean deg) {
        if (mrv) return MRV(deg);
        for (Item i : items) {
            if (!i.isAssigned) return i;
        }
        return null;
    }

    /**
     * Chooses the unassigned item from the given list with the fewest values in its domain.
     * In the case of a tie, pick the one with the most binary constraints with unassigned variables.
     * @return the item with minimum remaining values
     */
    private Item MRV(boolean deg) {
        ArrayList<Item> ties = new ArrayList<>();
        int minValues = Integer.MAX_VALUE;
        for (Item item : items) {
            if (!item.isAssigned) {
                if (item.domain.size() < minValues) { // new minimum
                    ties.clear();
                    ties.add(item);
                } else if (item.domain.size() == minValues) { // tied for minimum
                    ties.add(item);
                }
            }
        }
        if (!deg || ties.size() == 1) { return ties.get(0); }
        // break ties with maximum degree heuristic
        Item blankItem = new Item("", 0);
        int maxDegree = -1;
        for (Item item : ties) {
            int degree = degree(item);
            if (degree > maxDegree) {
                maxDegree = degree;
                blankItem = item;
            }
        }
        return blankItem;
    }

    /**
     * Finds the number of binary constraints the given unassigned item shares with
     * other unassigned items
     * @param i the item to find the degree of
     * @return the degree of i
     */
    private int degree(Item i) {
        int count = 0;
        for (Constraint c : i.constraints) {
            if (c instanceof AbstractBinaryConstraint) {
                if (!((AbstractBinaryConstraint) c).item1.isAssigned
                        && !((AbstractBinaryConstraint) c).item2.isAssigned) {
                    count++;
                }
            }
        }
        return count;
    }

    private ArrayList<Bag> orderBags(Item i, boolean lcv) {
        if (lcv) return LCV(i);
        return i.domain;
    }

    /**
     * Orders the domain of the item from least to most constraining values.
     * The least constraining value is the one that would eliminate
     * the fewest values from the domains of adjacent items, ie those
     * with which it shares a binary constraint
     * @param i the item whose values we want to order
     * @return list of possible values in order
     */
    private ArrayList<Bag> LCV(Item i) {
        int[] conflictTotals = new int[i.domain.size()];
        // go through each bag in the item's domain
        int index = 0;
        for (Bag b : i.domain) {
            b.addItem(i); // put the item in that bag
            // go through each binary constraint on the item
            for (Constraint c : i.constraints) {
                if (c instanceof AbstractBinaryConstraint) {
                    Item otherItem;
                    if (i.name == ((AbstractBinaryConstraint) c).item1.name) {
                        otherItem = ((AbstractBinaryConstraint) c).item2;
                    } else {
                        otherItem = ((AbstractBinaryConstraint) c).item1;
                    }
                    if (!otherItem.isAssigned) {
                        for (Bag b2 : otherItem.domain) { // go through each value in the other item's domain
                            b2.addItem(otherItem); // put the other item in that bag
                            if (!c.check(items, bags)) conflictTotals[index]++; // if it violates the constraint increment counter in parallel array
                            b2.removeItem(otherItem); // loop cleanup
                        }
                    }
                }
            }
            // loop cleanup
            b.removeItem(i);
            index++;
        }
        // use parallel array to order the item's domain
        ArrayList<Bag> result = new ArrayList<Bag>(i.domain.size());
        for (int pos = 0; pos < conflictTotals.length; pos++) { // pos = position in result
            // find minimum number of conflicts
            int min = Integer.MAX_VALUE;
            int minIndex = 0;
            for (int idx = 0; idx < conflictTotals.length; idx++) { // idx = index in conflictTotals
                if (conflictTotals[idx] < min) {
                    min = conflictTotals[idx];
                    minIndex = idx;
                }
            }
            conflictTotals[minIndex] = Integer.MAX_VALUE; // push up the minimum
            result.add(pos, i.domain.get(minIndex)); // add corresponding Bag from the item's domain
        }
        return result;
    }

    /**
     * Eliminates any values inconsistent with the value of the given item
     * from the domains of adjacent items, ie those with which it shares a
     * binary constraint
     * @param i the item to forward-check
     */
    private void forwardChecking(Item i) {
        // go through all the binary constraints
        for (Constraint c : i.constraints) {
            if (c instanceof AbstractBinaryConstraint) {
                // get the other item
                Item otherItem;
                if (i.name == ((AbstractBinaryConstraint) c).item1.name) {
                    otherItem = ((AbstractBinaryConstraint) c).item2;
                } else {
                    otherItem = ((AbstractBinaryConstraint) c).item1;
                }
                // go through the other item's domain
                for (Bag b2 : otherItem.domain) {
                    b2.addItem(otherItem);
                    // if this assignment violates the constraint, reduce the domain
                    if (!c.check(items, bags)) otherItem.domain.remove(b2);
                    b2.removeItem(otherItem); // loop cleanup
                }
            }
        }
    }

    /**
     * Checks if every item is assigned
     * @return true if every item is assigned
     */
    private boolean allAssigned() {
        for (Item i : items) {
            if (!i.isAssigned) return false;
        }
        return true;
    }
}
