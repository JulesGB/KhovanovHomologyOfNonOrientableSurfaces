class Diagram:
    def __init__(self, ck, linkComponents=""):
        self.ck = ck
        self.linkComponents = linkComponents
        self.initializeLinkComponents()
        self.componentNumber = self.initializeComponentNumber()
        self.strandNumber = self.initializeStrandNumber()
        self.n = len(ck)
        self.np = self.initializePositiveCrossingNumber()
        self.nn = self.initializeNegativeCrossingNumber()
    def initializeLinkComponents(self):
        if self.linkComponents == "":
            self.linkComponents = [1] * len(self.ck)
            for i in range(len(self.ck)):
                self.linkComponents[i] = [1] * 4
    def initializePositiveCrossingNumber(self):
        np = 0
        for i in range(len(self.ck)):
            if int(self.ck[i][3]) % int(self.strandNumber[self.linkComponents[i][3]-1]) < int(self.ck[i][1]) % int(self.strandNumber[self.linkComponents[i][1]-1]):
                np += 1
        return np
    def initializeNegativeCrossingNumber(self):
        nn = 0
        for i in range(len(self.ck)):
            if not(int(self.ck[i][3]) % int(self.strandNumber[self.linkComponents[i][3]-1]) < int(self.ck[i][1]) % int(self.strandNumber[self.linkComponents[i][1]-1])):
                nn += 1
        return nn
    def initializeComponentNumber(self):
        c = 0
        for i in range(len(self.linkComponents)):
            for j in range(len(self.linkComponents[i])):
                if c < self.linkComponents[i][j]:
                    c = self.linkComponents[i][j]
        return c
    def initializeStrandNumber(self):
        s = [0] * self.componentNumber
        for i in range(self.componentNumber):
            for j in range(len(self.ck)):
                for k in range(len(self.ck[j])):
                    if (s[i] < self.ck[j][k]) and (self.linkComponents[j][k] == i+1):
                        s[i] = self.ck[j][k]
        return s



class Smoothing:
    def __init__(self, diagram, bs, label=""):
        self.diagram = diagram
        self.ck = self.diagram.ck
        self.linkComponents = self.diagram.linkComponents
        self.linkComponentNumber = self.diagram.componentNumber
        self.n = len(self.ck)
        self.np = self.diagram.np
        self.nn = self.diagram.nn
        self.strandNumber = self.diagram.strandNumber
        self.bs = bs
        self.label = label
        self.smoothingComponents = []
        self.smoothingComponentsLinkComponents = []
        self.initializeSmoothingComponents()
        self.h = self.setHomologicalGrading()
        self.q = self.setQuantumGrading()
    def initializeSmoothingComponents(self):
        # create an array to track used strands
        # call it usedStrands[linkComponent, strand]
        usedStrands = [0] * self.linkComponentNumber
        for i in range(self.linkComponentNumber):
            usedStrands[i] = [0] * self.strandNumber[i]
        breaker = 0
        while not self.isAllOneArray(usedStrands):
            # find first strand for component
            # call it temp[linkComponent, strand]
            temp = [0, 0]
            breaker = 0
            for i in range(len(usedStrands)):
                for j in range(len(usedStrands[i])):
                    if usedStrands[i][j] == 0 and breaker == 0:
                        temp[0] = i + 1
                        temp[1] = j + 1
                        breaker = 1
            # if breaker=0 at this point, all strands have been used, so we are done
            if breaker != 0:
                firstStrand = temp[1]
                firstComponent = temp[0]
                # create string to track the strands in the smoothing
                component = str(firstStrand)
                # create string to track associated component for each strand in smoothing
                linkComponent = str(firstComponent)
                usedStrands[temp[0]-1][temp[1]-1] = 1
                nextStrand = 0
                nextComponent = 0
                direction = 0
                while nextStrand != firstStrand or nextComponent != firstComponent:
                    if nextStrand == 0:
                        nextStrand = firstStrand
                        nextComponent = firstComponent
                    nxt = self.findNextStrand(nextComponent, nextStrand, direction)
                    nextComponent = nxt[0]
                    nextStrand = nxt[1]
                    direction = nxt[2]
                    if nextStrand != firstStrand or nextComponent != firstComponent:
                        component = component + "," + str(nextStrand)
                        linkComponent = linkComponent + "," + str(nextComponent)
                        usedStrands[nxt[0]-1][nxt[1]-1] = 1
                self.smoothingComponents.append(component)
                self.smoothingComponentsLinkComponents.append(linkComponent)
    def findNextStrand(self, linkcomponent, strand, direction):
        nxt = []
        for i in range(self.n):
            if direction == 0:
                # Case 1: check if strand is first strand of crossing and verify direction (2 cases)
                if self.ck[i][0] == strand and self.linkComponents[i][0] == linkcomponent:
                    right = 1
                    left = 3
                    # Return array based on smoothing and direction of over-strand (2 cases)
                    if self.bs[i] == '0':
                        if int(self.ck[i][1]) % int(self.strandNumber[self.linkComponents[i][1]-1]) == (int(self.ck[i][3]) % int(self.strandNumber[self.linkComponents[i][3]-1]) + 1) % int(self.strandNumber[self.linkComponents[i][3]-1]):
                            nxt.append(self.linkComponents[i][right])
                            nxt.append(self.ck[i][right])
                            nxt.append(0)
                            return nxt
                        else:
                            nxt.append(self.linkComponents[i][right])
                            nxt.append(self.ck[i][right])
                            nxt.append(1)
                            return nxt
                    else:
                        if int(self.ck[i][1]) % int(self.strandNumber[self.linkComponents[i][1]-1]) == (int(self.ck[i][3]) % int(self.strandNumber[self.linkComponents[i][3]-1]) + 1) % int(self.strandNumber[self.linkComponents[i][3]-1]):
                            nxt.append(self.linkComponents[i][left])
                            nxt.append(self.ck[i][left])
                            nxt.append(1)
                            return nxt
                        else:
                            nxt.append(self.linkComponents[i][left])
                            nxt.append(self.ck[i][left])
                            nxt.append(0)
                            return nxt
                # Case 2: check if strand is second strand of crossing and verify direction
                if (self.ck[i][1] == strand) and self.linkComponents[i][1] == linkcomponent and int(self.ck[i][3]) % int(self.strandNumber[self.linkComponents[i][3]-1]) == (int(self.ck[i][1]) % int(self.strandNumber[self.linkComponents[i][1]-1]) + 1) % int(self.strandNumber[self.linkComponents[i][1]-1]):
                    right = 2
                    left = 0
                    # Return array based on smoothing (direction of under-strand is fixed)
                    if self.bs[i] == '0':
                        nxt.append(self.linkComponents[i][left])
                        nxt.append(self.ck[i][left])
                        nxt.append(1)
                        return nxt
                    else:
                        nxt.append(self.linkComponents[i][right])
                        nxt.append(self.ck[i][right])
                        nxt.append(0)
                        return nxt
                # Case 3: check if strand is fourth strand of crossing and verify direction
                if (self.ck[i][3] == strand) and self.linkComponents[i][3] == linkcomponent and int(self.ck[i][1]) % int(self.strandNumber[self.linkComponents[i][1]-1]) == (int(self.ck[i][3]) % int(self.strandNumber[self.linkComponents[i][3]-1]) + 1) % int(self.strandNumber[self.linkComponents[i][3]-1]):
                    right = 0
                    left = 2
                    # Return array based on smoothing (direction of under-strand is fixed)
                    if self.bs[i] == '0':
                        nxt.append(self.linkComponents[i][left])
                        nxt.append(self.ck[i][left])
                        nxt.append(0)
                        return nxt
                    else:
                        nxt.append(self.linkComponents[i][right])
                        nxt.append(self.ck[i][right])
                        nxt.append(1)
                        return nxt
            # otherwise the direction is against (direction = 0)
            else:
                # Case 4: check if strand is second strand of crossing and verify direction
                if (self.ck[i][1] == strand) and self.linkComponents[i][1] == linkcomponent and int(self.ck[i][1]) % int(self.strandNumber[self.linkComponents[i][1]-1]) == (int(self.ck[i][3]) % int(self.strandNumber[self.linkComponents[i][3]-1]) + 1) % int(self.strandNumber[self.linkComponents[i][3]-1]):
                    right = 2
                    left = 0
                    # Return array based on smoothing (direction of under-strand is fixed)
                    if self.bs[i] == '0':
                        nxt.append(self.linkComponents[i][left])
                        nxt.append(self.ck[i][left])
                        nxt.append(1)
                        return nxt
                    else:
                        nxt.append(self.linkComponents[i][right])
                        nxt.append(self.ck[i][right])
                        nxt.append(0)
                        return nxt
                # Case 5: check if strand is third strand of crossing and verify direction (2 cases)
                if (self.ck[i][2] == strand and self.linkComponents[i][2] == linkcomponent):
                    right = 3
                    left = 1
                    # Return array based on smoothing and direction of over-strand (2 cases)
                    if self.bs[i] == '0':
                        if int(self.ck[i][1]) % int(self.strandNumber[self.linkComponents[i][1]-1]) == (int(self.ck[i][3]) % int(self.strandNumber[self.linkComponents[i][3]-1]) + 1) % int(self.strandNumber[self.linkComponents[i][3]-1]):
                            nxt.append(self.linkComponents[i][right])
                            nxt.append(self.ck[i][right])
                            nxt.append(1)
                            return nxt
                        else:
                            nxt.append(self.linkComponents[i][right])
                            nxt.append(self.ck[i][right])
                            nxt.append(0)
                            return nxt
                    else:
                        if int(self.ck[i][1]) % int(self.strandNumber[self.linkComponents[i][1]-1]) == (int(self.ck[i][3]) % int(self.strandNumber[self.linkComponents[i][3]-1]) + 1) % int(self.strandNumber[self.linkComponents[i][3]-1]):
                            nxt.append(self.linkComponents[i][left])
                            nxt.append(self.ck[i][left])
                            nxt.append(0)
                            return nxt
                        else:
                            nxt.append(self.linkComponents[i][left])
                            nxt.append(self.ck[i][left])
                            nxt.append(1)
                            return nxt
                # Case 6: check if strand is fourth strand of crossing and verify direction
                if (self.ck[i][3] == strand) and self.linkComponents[i][3] == linkcomponent and int(self.ck[i][3]) % int(self.strandNumber[self.linkComponents[i][3]-1]) == (int(self.ck[i][1]) % int(self.strandNumber[self.linkComponents[i][1]-1]) + 1) % int(self.strandNumber[self.linkComponents[i][1]-1]):
                    right = 0
                    left = 2
                    # Return array based on smoothing (direction of under-strand is fixed)
                    if self.bs[i] == '0':
                        nxt.append(self.linkComponents[i][left])
                        nxt.append(self.ck[i][left])
                        nxt.append(0)
                        return nxt
                    else:
                        nxt.append(self.linkComponents[i][right])
                        nxt.append(self.ck[i][right])
                        nxt.append(1)
                        return nxt
    def isAllOneArray(self, arr):
        test = True
        for i in range(len(arr)):
            for j in range(len(arr[i])):
                test = test and (arr[i][j] == 1)
        return test
    def setHomologicalGrading(self):
        h = 0
        for i in range(len(self.bs)):
            h = h + int(self.bs[i])
        return h - self.nn
    def setQuantumGrading(self):
        vp = 0
        vn = 0
        for i in range(len(self.label)):
            if self.label[i] == '1':
                vp += 1
            else:
                vn += 1
        return vp - vn + self.h + self.np - self.nn
    def setLabel(self, lab):
        if len(lab) == len(self.smoothingComponents):
            self.label = lab
            self.q = self.setQuantumGrading()
        else:
            raise Exception("Label size does not match number of components.")
    def getLabel(self):
        return self.label
    def getComponents(self):
        return self.smoothingComponents
    def getBinarySequence(self):
        return self.bs
    def printSmoothing(self):
        print("Smoothing: " + self.bs)
        print("   components:", self.smoothingComponents)
        print("   link components:", self.smoothingComponentsLinkComponents)
        print("   label:", self.label)
        print("   grading: (" + str(self.h) + "," + str(self.q) + ")")
        print("-----")




class Cycle:
    def __init__(self, diagram, smoothings, coefficients):
        self.diagram = diagram
        self.sm = smoothings
        self.c = coefficients
    def addSmoothing(self, smoo, coeff):
        self.sm.append(smoo)
        self.c.append(coeff)
    def getHomologicalGrading(self):
        return self.sm[0].h
    def getQuantumGrading(self):
        return self.sm[0].q
    def getCoefficients(self):
        return self.c
    def getSmoothings(self):
        return self.sm
    def printCycle(self):
        print("------")
        print("Cycle:")
        print("------")
        print(" bigrading: (" + str(self.sm[0].h) + "," + str(self.sm[0].q) + ")")
        for i in range(len(self.sm)):
            self.sm[i].printSmoothing()



class ChainComplex:
    def __init__(self, diagram):
        self.diagram = diagram
        self.bs = self.initializeBinarySequences()
        self.chainElements = self.initializeSmoothings()
    def initializeBinarySequences(self):
        binarySequences = []
        for i in range(2**self.diagram.n):
            temp = bin(i).replace("0b", "")
            while len(temp) < self.diagram.n:
                temp = '0' + temp
            binarySequences.append(temp)
        return binarySequences
    def initializeSmoothings(self):
        smoothings = []
        for i in range(len(self.bs)):
            binarySequence = self.bs[i]
            smooth = Smoothing(self.diagram, binarySequence)
            for j in range(2**len(smooth.smoothingComponents)):
                smoothings.append(Smoothing(self.diagram, binarySequence, self.addPadding(bin(j).replace("0b", ""), len(smooth.smoothingComponents)).replace("0", "x")))
        return smoothings
    @staticmethod
    def addPadding(temp, num):
        difference = num - len(temp)
        for i in range(difference):
            temp = "0" + temp
        return temp
    def d(self, smoo):
        dsmoothings = []
        dcoefficients = []
        dsmoo = Cycle(self.diagram, dsmoothings, dcoefficients)
        binarySequence = smoo.bs
        for i in range(len(binarySequence)):
            if binarySequence[i] == '0':
                j = i+1
                newBinarySequence = binarySequence[:i] + '1' + binarySequence[j:]
                newSmoothing = Smoothing(self.diagram, newBinarySequence)
                xi = 0
                for j in range(i):
                    xi = xi + int(binarySequence[j])
                newCoefficient = (-1)**xi
                oldComponents = smoo.smoothingComponents
                newComponents = newSmoothing.smoothingComponents
                reducedOldComponents = []
                reducedNewComponents = []
                oldLabel = smoo.label
                reducedOldLabel = ""
                reducedNewLabel = ""
                for j in range(len(newComponents)):
                    if newComponents[j] not in oldComponents:
                        reducedNewComponents.append(newComponents[j])
                for j in range(len(oldComponents)):
                    if oldComponents[j] not in newComponents:
                        reducedOldComponents.append(oldComponents[j])
                        reducedOldLabel = reducedOldLabel + oldLabel[j]
                if len(reducedOldComponents) > len(reducedNewComponents):
                    if reducedOldLabel[0] == '1' and reducedOldLabel[1] == '1':
                        reducedNewLabel = '1'
                    elif (reducedOldLabel[0] == '1' and reducedOldLabel[1] == 'x') or (reducedOldLabel[1] == '1' and reducedOldLabel[0] == 'x'):
                        reducedNewLabel = 'x'
                    else:
                        newCoefficient = 0
                elif len(reducedOldComponents) < len(reducedNewComponents):
                    if reducedOldLabel == '1':
                        reducedNewLabel = '1x+x1'
                    elif reducedOldLabel == 'x':
                        reducedNewLabel = 'xx'
                    else:
                        raise Exception("Impossible label on component when performing differential.")
                else:
                    raise Exception("Error in number of components before/after differential.")
                newLabel = ""
                count = 0
                newLabel1 = ""
                newLabel2 = ""
                for j in range(len(newComponents)):
                    if newComponents[j] in oldComponents:
                        if count == 0:
                            newLabel = newLabel + oldLabel[oldComponents.index(newComponents[j])]
                        else:
                            newLabel1 = newLabel1 + oldLabel[oldComponents.index(newComponents[j])]
                            newLabel2 = newLabel2 + oldLabel[oldComponents.index(newComponents[j])]
                    else:
                        if len(reducedNewLabel) == 1:
                            newLabel = newLabel + reducedNewLabel
                        elif reducedNewLabel == 'xx':
                            newLabel = newLabel + 'x'
                        else:
                            if count == 0:
                                newLabel1 = newLabel + '1'
                                newLabel2 = newLabel + 'x'
                                count += 1
                            else:
                                newLabel1 = newLabel1 + 'x'
                                newLabel2 = newLabel2 + '1'
                if newCoefficient != 0:
                    if newLabel1 != "":
                        newSmoothing2 = Smoothing(self.diagram, newBinarySequence)
                        newCoefficient2 = newCoefficient
                        newSmoothing.setLabel(newLabel1)
                        dsmoo.addSmoothing(newSmoothing, newCoefficient)
                        newSmoothing2.setLabel(newLabel2)
                        dsmoo.addSmoothing(newSmoothing2, newCoefficient2)
                    else:
                        newSmoothing.setLabel(newLabel)
                        dsmoo.addSmoothing(newSmoothing, newCoefficient)
        return dsmoo
    def getDElement(self, cy):
        delement = []
        delementCoe = []
        for i in range(len(cy.sm)):
            nextSmoothing = cy.sm[i]
            cycle = self.d(nextSmoothing)
            dsmoo = cycle.sm
            dcoe = cycle.c
            for j in range(len(dcoe)):
                temp = dcoe[j]
                dcoe[j] = temp*cy.c[i]
            temp = 0
            for j in range(len(dsmoo)):
                for k in range(len(delement)):
                    if self.compareSmoothings(dsmoo[j], delement[k]):
                        delementCoe[k] = delementCoe[k] + dcoe[j]
                        temp = 1
                if temp != 1:
                    delement.append(dsmoo[j])
                    delementCoe.append(dcoe[j])
                    temp = 0
        return Cycle(self.diagram, delement, delementCoe)
    def isCycle(self, cy):
        delement = []
        delementCoe = []
        for i in range(len(cy.sm)):
            nextSmoothing = cy.sm[i]
            cycle = self.d(nextSmoothing)
            dsmoo = cycle.sm
            dcoe = cycle.c
            for j in range(len(dcoe)):
                temp = dcoe[j]
                dcoe[j] = temp*cy.c[i]
            temp = 0
            for j in range(len(dsmoo)):
                for k in range(len(delement)):
                    if self.compareSmoothings(dsmoo[j], delement[k]):
                        delementCoe[k] = delementCoe[k] + dcoe[j]
                        temp = 1
                    if temp != 1:
                        delement.append(dsmoo[j])
                        delementCoe.append(dcoe[j])
                        temp = 0
            isCycle = True
            for j in range(len(delementCoe)):
                isCycle = isCycle and delementCoe[j] == 0
            return isCycle
    def getDomain(self, hom, qua):
        domain = []
        for i in range(len(self.chainElements)):
            if self.chainElements[i].h == hom and self.chainElements[i].q == qua:
                domain.append(self.chainElements[i])
        return domain
    def getImage(self, hom, qua):
        domain = self.getDomain(hom, qua)
        image = []
        for i in range(len(domain)):
            imageElement = self.d(domain[i])
            imageSmoothings = imageElement.getSmoothings()
            if len(imageSmoothings) != 0:
                for j in range(len(imageSmoothings)):
                    if not self.isInImage(image, imageSmoothings[j]):
                        image.append(imageSmoothings[j])
        return image
    def getChainMap(self, hom, qua):
        domain = self.getDomain(hom, qua)
        image = self.getImage(hom, qua)
        return self.initializeMatrix(domain, image)
    def initializeMatrix(self, dom, image):
        matrix = [0] * len(image)
        for i in range(len(image)):
            matrix[i] = [0] * len(dom)
        for i in range(len(dom)):
            cycle = self.d(dom[i])
            smoos = cycle.getSmoothings()
            for j in range(len(image)):
                for k in range(len(smoos)):
                    if self.compareSmoothings(smoos[k], image[j]):
                        matrix[j][i] = cycle.c[k]
        return matrix
    def printChainGroup(self, hom, qua):
        print("Chain group:")
        for i in range(len(self.chainElements)):
            if self.chainElements[i].h == hom and self.chainElements[i].q == qua:
                self.chainElements[i].printSmoothing()
        print("-----")
    def printImage(self, hom, qua):
        print("Image:")
        domain = self.getDomain(hom, qua)
        image = []
        for i in range(len(domain)):
            imageElement = self.d(domain[i])
            imageSmoothings = imageElement.sm
            if len(imageSmoothings) != 0:
                for j in range(len(imageSmoothings)):
                    if not self.isInImage(image, imageSmoothings[j]):
                        image.append(imageSmoothings[j])
        for i in range(len(image)):
            image[i].printSmoothing()
    def isInImage(self, arr, smoo):
        if len(arr) == 0:
            return False
        for i in range(len(arr)):
            if self.compareSmoothings(arr[i], smoo):
                return True
        return False
    @staticmethod
    def compareSmoothings(s1, s2):
        if s1.smoothingComponents == s2.smoothingComponents and s1.label == s2.label and s1.smoothingComponentsLinkComponents == s2.smoothingComponentsLinkComponents:
            return True
        return False



class KJClass:
    def __init__(self, diagram, cycle):
        print("...initializing diagram...")
        self.diagram = diagram
        print("Diagram initialized...")
        print("...initializing cycle...")
        self.cycle = cycle
        print("Cycle initialized...")
        print("...initializing chain complex...")
        self.ckh = ChainComplex(diagram)
        print("Chain complex initialized...")
        print("...initializing class...")
        self.h = self.cycle.getHomologicalGrading()
        self.q = self.cycle.getQuantumGrading()
        self.domain = self.ckh.getDomain(self.h-1, self.q)
        self.image = self.ckh.getImage(self.h-1, self.q)
        print("Class initialized...")
    def classAsVector(self):
        kj = [0] * len(self.image)
        for i in range(len(self.cycle.sm)):
            for j in range(len(self.image)):
                if self.ckh.compareSmoothings(self.cycle.sm[i], self.image[j]):
                    kj[j] = kj[j] + self.cycle.c[i]
        return kj
    def isCycle(self):
        print("...checking if class is a cycle...")
        if self.ckh.isCycle(self.cycle):
            print("Class is a cycle.")
    def getMap(self):
        return self.ckh.getChainMap(self.h-1, self.q)
    def printMap(self):
        m = self.getMap()
        for i in range(len(m)):
            for j in range(len(m[i])):
                print(m[i][j])
    def buildMatrix(self):
        print("...initializing matrix...")
        mat = matrix(ZZ, self.getMap())
        print("Matrix initialized...")
        return mat
    def buildVector(self):
        return vector(ZZ, self.classAsVector())
    def isNontrivial(self):
        try:
            mat = self.buildMatrix()
            print("...checking nontriviality of class...")
            w = mat.solve_right(self.buildVector())
            print("Solution is below (could be trivial!)")
            print(w)
        except ValueError:
            print("Class is nontrivial (solve_right threw an error).")